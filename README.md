# DriftSE

This repository provides the official implementation of the following paper:

**Speech Enhancement Based on Drifting Models (Interspeech 2026, Oral Presentation)**
Liang Xu, Diego Caviedes-Nozal, W. Bastiaan Kleijn, Longfei Felix Yan, Rasmus Kongsgaard Olsson
*Interspeech 2026*

🔗 [**Project Website**](https://liangxu123.github.io/driftse/)  |  📄 [**arXiv Preprint**](https://arxiv.org/abs/2604.24199)

---

## 📖 Highlights

- **Novel Generative Paradigm:** Formulates speech enhancement as a distributional equilibrium problem, eliminating the need for iterative denoising or trajectory-based sampling.
- **Native One-Step Inference:** Achieves single-step (1 NFE) enhancement by evolving the pushforward distribution of a mapping function to directly match the clean speech distribution via a Drifting Field.
- **Semantic Latent Drifting:** Operates in a hierarchical self-supervised speech latent space (HuBERT, WavLM, DistilHuBERT), providing rich and stable training signals that capture both acoustic and phonetic structure.
- **Unpaired Learning:** Natively supports training on fully unpaired noisy/clean speech data, enabling cross-dataset and cross-gender generalization without paired supervision.
- **State-of-the-Art Generalization:** Achieves state-of-the-art WV-MOS and SCOREQ on the DNS Challenge 2020 blind test set, outperforming multi-step diffusion and consistency-based baselines.

---

## 📊 Performance Benchmark

### VoiceBank-DEMAND (VB-DMD) — In-Domain Evaluation

| Method                                   |     NFE     |   PESQ (↑)   |   SI-SDR (↑)   |   ESTOI (↑)   |  DNSMOS (↑)  |  SCOREQ (↑)  |
| ---------------------------------------- | :---------: | :------------: | :-------------: | :------------: | :------------: | :------------: |
| MetricGAN+                               |      1      |      3.13      |      8.50      |      0.83      |      3.22      |      3.82      |
| UNIVERSE++                               |      8      |      2.91      |      18.00      |      0.85      |      3.45      | **4.35** |
| SGMSE+                                   |     30     |      2.90      |      16.90      |      0.85      |      3.48      |      3.98      |
| ROSE-CD                                  |      1      |      3.49      |      17.80      |      0.87      |      3.49      |      4.23      |
| SBCTM                                    |      1      | **3.56** |      12.70      |      0.87      |      3.55      | **4.35** |
| MeanFlowSE                               |      1      |      2.81      | **19.97** | **0.88** | **3.58** |      4.25      |
| *DriftSE (WavLM, L24)*                 |      1      |      2.90      |      12.60      |      0.84      |      3.36      |      3.93      |
| *DriftSE (WavLM)*                      |      1      |      3.03      |      14.00      |      0.85      | **3.54** | **4.17** |
| *DriftSE (HuBERT)*                     |      1      |      2.94      |      12.50      |      0.84      |      3.49      |      4.14      |
| *DriftSE (DistilHuBERT)*               |      1      |      3.00      |      15.60      |      0.85      |      3.48      |      4.15      |
| ***DriftSE† (DistilHuBERT)†*** | **1** | **3.45** | **20.60** | **0.87** |      3.49      |      4.11      |
| *DriftSE (Unpaired, map to DNS)*       |      1      |      2.00      |      6.60      |      0.74      | **3.61** | **3.92** |

> † Jointly trained with auxiliary PESQ and SI-SDR losses.
> *Italic rows*: DriftSE ablation and unpaired variants.

### DNS Challenge 2020 Blind Test Set — Real-World Generalization

| Method                                 |     NFE     |  WV-MOS (↑)  |  SCOREQ (↑)  |    SIG (↑)    |    BAK (↑)    |   OVRL (↑)   |
| -------------------------------------- | :---------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| MetricGAN+                             |      1      |      1.23      |      2.08      |      3.28      |      3.45      |      2.70      |
| UNIVERSE++                             |      8      |      1.99      |      2.27      |      3.45      |      3.52      |      2.93      |
| SGMSE+                                 |     30     |      2.34      | **2.95** | **4.12** | **3.94** | **3.62** |
| ROSE-CD                                |      1      | **2.37** |      2.81      |      4.01      |      3.80      |      3.42      |
| SBCTM                                  |      1      |      2.24      |      2.78      |      3.83      |      3.88      |      3.33      |
| MeanFlowSE                             |      1      |      2.20      |      2.79      |      3.88      |      3.51      |      3.21      |
| *DriftSE (WavLM)*                    |      1      |      2.62      |      2.67      |      3.85      | **3.94** | **3.42** |
| *DriftSE (HuBERT)*                   |      1      |      2.56      |      2.74      | **3.92** |      3.79      |      3.40      |
| ***DriftSE (DistilHuBERT)†*** | **1** | **2.65** | **2.97** |      3.78      |      3.84      |      3.31      |

---

## ⚙️ Installation & Setup

We recommend utilizing an isolated virtual environment with Python 3.11. To initialize the environment and install dependencies, execute:

```bash
# Clone the repository
git clone https://github.com/liangxu123/driftse.git
cd driftse

# Install required packages
pip install -r requirements.txt
```

*Note: For experiment tracking via Weights & Biases (W&B), please configure your environment using `wandb login` prior to initiating training. By default, `train.sh` sets `WANDB_MODE=dryrun` to run offline.*

---

## 🗄️ Dataset Preparation

Our data preprocessing pipeline is adapted from the [SGMSE+](https://github.com/sp-uhh/sgmse) framework. To configure the dataset directories, update the corresponding paths in `path_config.sh`. By default, the configuration points to the VoiceBank-DEMAND corpus paths.

Training uses **dynamic mixing**: 10,802 clean VoiceBank utterances are mixed on-the-fly with 18 DEMAND noise types at SNRs sampled from {0, 5, 10, 15} dB. Evaluation is performed on the standard pre-mixed VB-DMD test set (824 utterances).

---

## 🤗 SSL Encoder Checkpoints

DriftSE uses frozen self-supervised speech encoders (WavLM-Large, HuBERT-Large, DistilHuBERT) to compute the latent drifting field. These must be downloaded **before** training.

**Download:** [Google Drive — `latent_ckpt/`](https://drive.google.com/file/d/1NFW91B7jwwJV4dUtcOZyaaHyihjBiuKc/view?usp=sharing)

The folder contains three sub-directories:

```
latent_ckpt/
├── wavlm-large-local/       # WavLM-Large (1024-d, 24 layers)
├── hubert-large-local/      # HuBERT-Large (1024-d, 24 layers)
└── distilhubert-local/      # DistilHuBERT (768-d, 2 layers)
```

Extract and place the downloaded folder at the **root of this repository** so the paths resolve to `./latent_ckpt/`:

```bash
# After downloading, your directory should look like:
ls ./latent_ckpt/
# wavlm-large-local  hubert-large-local  distilhubert-local
```

The encoder paths are configured in `path_config.sh` and default to `./latent_ckpt/`. You can override them by editing that file.

**Verify the setup** by running the encoder test script:

```bash
python test_encoder.py
```

Expected output:
```
[WavLM]         Layer  6 shape: torch.Size([1, 49, 1024])
[WavLM]         Layer 12 shape: torch.Size([1, 49, 1024])
[WavLM]         Layer 24 shape: torch.Size([1, 49, 1024])

[HuBERT]        Layer  6 shape: torch.Size([1, 49, 1024])
[HuBERT]        Layer 12 shape: torch.Size([1, 49, 1024])
[HuBERT]        Layer 24 shape: torch.Size([1, 49, 1024])

[DistilHuBERT]  Total hidden states: 3
[DistilHuBERT]  Layer  0 shape: torch.Size([1, 49, 768])
[DistilHuBERT]  Layer  1 shape: torch.Size([1, 49, 768])
[DistilHuBERT]  Layer  2 shape: torch.Size([1, 49, 768])
```

---

## 🚀 Training

To train a DriftSE model, execute:

```bash
bash ./train.sh <GPU_ID> [CONFIG_PATH]
# e.g., bash ./train.sh 0
# e.g., bash ./train.sh 0 ./config/with_z/v2_drifteight_distillhubert_three_layers.json
```

- **`GPU_ID`** (required): The GPU index to use for training.
- **`CONFIG_PATH`** (optional): Path to a JSON config file. Defaults to `./config/with_z/v2_drifteight_distillhubert_three_layers.json`.

The repository provides two core DriftSE formulations, controlled via config:

- **Direct Mapping (`no_z/`)**: Deterministic mapping `f_θ(y)` from noisy to clean speech. Set `σ=0` for highest PESQ/SI-SDR fidelity.
- **Conditional Generator (`with_z/`)**: Stochastic mapping `f_θ(ε, y)` from Gaussian noise conditioned on noisy speech, for higher perceptual quality (DNSMOS, SCOREQ).

**Implementation details:**

- **Backbone**: NCSN++V2 without time embedding
- **Audio**: 16 kHz, STFT with 510-point Hann window, hop length 128, spectral compression
- **SSL Encoder**: Pre-trained HuBERT-Large, WavLM-Large, or DistilHuBERT (all frozen)
- **Kernel**: Multi-temperature exponential kernel with τ ∈ {0.1, 0.5, 1.0}
- **Optimizer**: AdamW, lr=5×10⁻⁴, weight decay=0.01, batch size=16, 100 epochs

---

## 📈 Evaluation

To evaluate a trained model and compute all objective metrics, run:

```bash
bash ./test.sh <GPU_ID> [CONFIG_PATH]
# e.g., bash ./test.sh 0
# e.g., bash ./test.sh 0 ./config/with_z/v2_drifteight_distillhubert_three_layers.json
```

The evaluation pipeline runs two sequential phases:

1. **Enhancement**: Generates enhanced audio via `enhancement.py`
2. **Objective Metrics**: Computes PESQ, ESTOI, SI-SDR via `calc_metrics.py` (using paths from `path_config.sh`)

---

## 🔗 Pre-trained Checkpoints & Enhanced Audio

We release the pre-trained model checkpoint and the corresponding enhanced audio outputs to facilitate reproducibility.

| Resource | Description | Link |
|---|---|---|
| **Checkpoint** (`logs/`) | Trained DriftSE model weights | [Google Drive](https://drive.google.com/file/d/1ekzJQidIojhjlj6oaUzQBKp4Pil6jIz7/view?usp=sharing) |
| **Enhanced Audio** (`out/`) | Enhanced VB-DMD test set outputs | [Google Drive](https://drive.google.com/file/d/1xdfUnp6Pc02Ug137dCPTvyNc3uYjO0dh/view?usp=sharing) |

**Usage:** Download and extract each archive into the repository root so the paths match the config defaults:

```bash
# Checkpoint → ./logs/
# Enhanced audio → ./out/
ls ./logs/   # distillhubert_three_layers_with_z/last.ckpt, ...
ls ./out/    # distillhubert_three_layers_with_z/
```

---

## 📝 Citation

If this codebase or methodology proves useful in your research, please cite:

```bibtex
@inproceedings{xu2026driftse,
  author    = {Liang Xu and Diego Caviedes-Nozal and W. Bastiaan Kleijn and Longfei Felix Yan and Rasmus Kongsgaard Olsson},
  title     = {Speech Enhancement Based on Drifting Models},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026}
}
```

---

## 🙏 Acknowledgments

We express our gratitude to the authors of the [SGMSE+](https://github.com/sp-uhh/sgmse) repository, upon whose foundational work this codebase is built. We also thank the authors of [Drifting Models](https://github.com/tyfeld/drifting-model) for the generative framework that inspired this work.
