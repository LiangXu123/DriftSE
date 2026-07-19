import os
import sys
import warnings
from os.path import join
from glob import glob
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import librosa

# Allow importing from parent directory when run from test_script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pystoi import stoi
from other import energy_ratios, mean_std

warnings.filterwarnings("ignore", category=UserWarning)

N_WORKERS = max(1, os.cpu_count() - 2) if os.cpu_count() else 1

def build_file_index(root_dir, ext=".wav"):
    """Return {basename: absolute_path} for every wav under root_dir."""
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(ext):
                index[fn] = os.path.join(dirpath, fn)
    return index

def resolve_clean(rel_path, clean_dir, clean_index):
    base_name = os.path.basename(rel_path)
    candidates = [base_name]
    if '_' in base_name:
        candidates.append(base_name.split('_')[0] + '.wav')
        candidates.append("_".join(base_name.split('_')[:-1]) + ".wav")

    for cb in candidates:
        clean_rel = os.path.join(os.path.dirname(rel_path), cb)
        for cand in [join(clean_dir, clean_rel), join(clean_dir, cb)]:
            if os.path.exists(cand):
                return cand
        if cb in clean_index:
            return clean_index[cb]
    return None

def resolve_enhanced(rel_path, enhanced_dir, enhanced_index):
    base_name = os.path.basename(rel_path)
    for cand in [join(enhanced_dir, rel_path),
                 join(enhanced_dir, base_name),
                 join(enhanced_dir, rel_path.replace("/", ""))]:
        if os.path.exists(cand):
            return cand
    return enhanced_index.get(base_name, None)

def process_file_parallel(args_tuple):
    """Compute PESQ / STOI / ESTOI / SI-SDR in a worker process."""
    rel_path, clean_file, noisy_file, enhanced_file = args_tuple

    try:
        x,     sr_x     = read(clean_file)
        y,     _        = read(noisy_file)
        x_hat, sr_x_hat = read(enhanced_file)
    except Exception as e:
        return None, f"Read error for {rel_path}: {e}"

    min_len = min(len(x), len(y), len(x_hat))
    x     = x[:min_len]
    y     = y[:min_len]
    x_hat = x_hat[:min_len]
    n     = y - x

    x_16k     = librosa.resample(x,     orig_sr=sr_x,     target_sr=16000) if sr_x     != 16000 else x
    x_hat_16k = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat
    min_16k   = min(len(x_16k), len(x_hat_16k))
    x_16k, x_hat_16k = x_16k[:min_16k], x_hat_16k[:min_16k]

    pesq_s   = pesq(16000, x_16k, x_hat_16k, 'wb')
    stoi_s   = stoi(x, x_hat, sr_x, extended=False)
    estoi_s  = stoi(x, x_hat, sr_x, extended=True)
    si_sdr_s = energy_ratios(x_hat, x, n)[0]

    row = {
        "filename":      rel_path,
        "pesq":          pesq_s,
        "stoi":          stoi_s,
        "estoi":         estoi_s,
        "si_sdr":        si_sdr_s,
    }
    return row, None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True,
                        help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True,
                        help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True,
                        help='Directory containing the enhanced data')
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    print("Indexing directories...")
    clean_index    = build_file_index(args.clean_dir)
    enhanced_index = build_file_index(args.enhanced_dir)

    noisy_files = sorted(set(
        glob(join(args.noisy_dir, '*.wav')) +
        glob(join(args.noisy_dir, '**', '*.wav'), recursive=True)
    ))

    tasks, skipped = [], 0
    for noisy_file in noisy_files:
        rel_path = os.path.relpath(noisy_file, args.noisy_dir)
        if not rel_path or rel_path == ".":
            continue
        clean_file    = resolve_clean(rel_path, args.clean_dir, clean_index)
        enhanced_file = resolve_enhanced(rel_path, args.enhanced_dir, enhanced_index)
        if not clean_file or not enhanced_file:
            skipped += 1
            continue
        tasks.append((rel_path, clean_file, noisy_file, enhanced_file))

    print(f"Found {len(tasks)} file pairs ({skipped} skipped).")

    print(f"\nComputing PESQ / STOI / ESTOI / SI-SDR ({args.workers} workers)...")
    rows, errors = [], []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_parallel, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            row, err = future.result()
            if err:
                errors.append(err)
            elif row:
                rows.append(row)

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:5]:
            print(" ", e)

    df = pd.DataFrame(rows)

    lines = [
        f"Evaluated {len(df)} files",
        "PESQ:         {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())),
        "STOI:         {:.4f} ± {:.4f}".format(*mean_std(df["stoi"].to_numpy())),
        "ESTOI:        {:.4f} ± {:.4f}".format(*mean_std(df["estoi"].to_numpy())),
        "SI-SDR:       {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())),
    ]

    print("\n".join(lines))

    with open(join(args.enhanced_dir, "_avg_results.txt"), "w") as log:
        log.write("\n".join(lines) + "\n")

    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)