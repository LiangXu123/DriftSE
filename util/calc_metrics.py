from os.path import join
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import librosa
import os

from pystoi import stoi
from other import energy_ratios, mean_std
import scoreq

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Predict quality of natural speech in NR mode
nr_scoreq = scoreq.Scoreq(data_domain='natural', mode='nr')
# Predict quality of natural speech in REF mode
ref_scoreq = scoreq.Scoreq(data_domain='natural', mode='ref')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True,
                        help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True,
                        help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True,
                        help='Directory containing the enhanced data')
    args = parser.parse_args()

    # Removed "moslqo" from dictionary initialization
    data = {"filename": [], "pesq": [], "stoi": [], "estoi": [],
            "si_sdr": [], "nr_scoreq": [], "ref_scoreq": []}

    # Evaluate standard metrics
    noisy_files = []
    # Using simple glob to avoid potential duplication if **/*.wav matches *.wav
    # If your directory structure is flat, just use *.wav. If recursive, use **/*.wav
    # Combining them might cause duplicates depending on the OS/glob version, 
    # but keeping your logic to be safe:
    noisy_files_1 = sorted(glob(join(args.noisy_dir, '*.wav')))
    noisy_files_2 = sorted(glob(join(args.noisy_dir, '**', '*.wav')))
    noisy_files = list(set(noisy_files_1 + noisy_files_2)) # Remove duplicates

    for noisy_file in tqdm(noisy_files):
        # filename = noisy_file.replace(args.noisy_dir, "")[1:]
        filename = noisy_file.replace(args.noisy_dir, "")
        filename = str(filename).replace("/", "")
        
        # Handle cases where filename might be empty after replace if paths overlap perfectly
        if not filename: continue 

        if 'dB' in filename:
            clean_filename = filename.split("_")[0] + ".wav"
        else:
            clean_filename = filename
        
        try:
            x, sr_x = read(join(args.clean_dir, clean_filename))
            y, sr_y = read(join(args.noisy_dir, filename))
            x_hat, sr_x_hat = read(join(args.enhanced_dir, filename))
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue

        # Ensure same length for all arrays
        min_len = min(len(x), len(y), len(x_hat))
        x = x[:min_len]
        y = y[:min_len]
        x_hat = x_hat[:min_len]

        # assert sr_x == sr_y == sr_x_hat
        n = y - x

        # Resample to 16kHz for PESQ if needed
        x_hat_16k = librosa.resample(
            x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat
        x_16k = librosa.resample(
            x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x

        min_len_16k = min(len(x_16k), len(x_hat_16k))
        x_16k = x_16k[:min_len_16k]
        x_hat_16k = x_hat_16k[:min_len_16k]

        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x_16k, x_hat_16k, 'wb'))
        
        # --- VISQOL CALCULATION DISABLED ---
        # data["moslqo"].append(visqol_api.Measure(x_16k, x_hat_16k).moslqo)
        
        data["stoi"].append(
            stoi(x, x_hat, sr_x, extended=False))
        data["estoi"].append(
            stoi(x, x_hat, sr_x, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])

        # Scoreq metrics
        data["nr_scoreq"].append(nr_scoreq.predict(
            test_path=join(args.enhanced_dir, filename), ref_path=None))
        data["ref_scoreq"].append(ref_scoreq.predict(test_path=join(
            args.enhanced_dir, filename), ref_path=join(args.clean_dir, clean_filename)))

    # Save results as DataFrame
    df = pd.DataFrame(data)

    # Print results
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    # print("MOSLQO(ViSQOL): {:.2f} ± {:.2f}".format(*mean_std(df["moslqo"].to_numpy())))
    print("STOI: {:.4f} ± {:.4f}".format(*mean_std(df["stoi"].to_numpy())))
    print("ESTOI: {:.4f} ± {:.4f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("NR_Scoreq: {:.2f} ± {:.2f}".format(
        *mean_std(df["nr_scoreq"].to_numpy())))
    print("REF_Scoreq: {:.2f} ± {:.2f}".format(
        *mean_std(df["ref_scoreq"].to_numpy())))

    # Save average results to file
    log = open(join(args.enhanced_dir, "_avg_results.txt"), "w")
    log.write("PESQ: {:.2f} ± {:.2f}".format(
        *mean_std(df["pesq"].to_numpy())) + "\n")
    # log.write("MOSLQO(ViSQOL): {:.2f} ± {:.2f}".format(*mean_std(df["moslqo"].to_numpy())) + "\n")
    log.write("STOI: {:.4f} ± {:.4f}".format(
        *mean_std(df["stoi"].to_numpy())) + "\n")
    log.write("ESTOI: {:.4f} ± {:.4f}".format(
        *mean_std(df["estoi"].to_numpy())) + "\n")
    log.write("SI-SDR: {:.1f} ± {:.2f}".format(*
        mean_std(df["si_sdr"].to_numpy())) + "\n")
    log.write("NR_Scoreq: {:.2f} ± {:.2f}".format(*
        mean_std(df["nr_scoreq"].to_numpy())) + "\n")
    log.write("REF_Scoreq: {:.2f} ± {:.2f}".format(*
        mean_std(df["ref_scoreq"].to_numpy())) + "\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)