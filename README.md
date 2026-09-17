# Speech Enhancement Based on Drifting Models

[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20arXiv-2604.24199-red.svg)](https://arxiv.org/abs/2604.24199)
[![github](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/liangxu123/driftse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Interspeech 2026](https://img.shields.io/badge/Interspeech%202026-Oral-blue.svg)]()

> **Speech Enhancement Based on Drifting Models** (Interspeech 2026, Oral Presentation)
>
> *Liang Xu, Diego Caviedes-Nozal, W. Bastiaan Kleijn, Longfei Felix Yan, Rasmus Kongsgaard Olsson*

🔗 [**Project Website**](https://liangxu123.github.io/driftse/) &nbsp;|&nbsp; 📄 [**arXiv Paper**](https://arxiv.org/abs/2604.24199) &nbsp;|&nbsp; 🤗 [**Hugging Face Space**](https://huggingface.co/spaces/LIANGXU123/DriftSE)

# DriftSE: Speech Enhancement with Generative Drifting

[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20arXiv-2609.12252-red.svg)](https://arxiv.org/abs/2609.12252)
[![github](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/LiangXu123/DriftSE/tree/dual-latent-DriftSE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---
> **DriftSE: Speech Enhancement with Generative Drifting** (Submitted to IEEE/ACM TASLP for possible publication. )
>
> *Liang Xu, Diego Caviedes-Nozal, W. Bastiaan Kleijn, Longfei Felix Yan, Rasmus Kongsgaard Olsson*
🔗 [**Project Website**](https://liangxu123.github.io/driftse/) &nbsp;|&nbsp; 📄 [**arXiv Paper**](https://arxiv.org/abs/2609.12252) &nbsp;|&nbsp
> 
## ✨ Key Highlights

- **Novel Generative Paradigm** — Formulates speech enhancement as a distributional equilibrium problem, eliminating the need for iterative denoising or trajectory-based sampling.
- **Native One-Step Inference** — Achieves single-step (1 NFE) enhancement by evolving the pushforward distribution of a mapping function to directly match the clean speech distribution via a Drifting Field.
- **Semantic Latent Drifting** — Operates in a hierarchical self-supervised speech latent space (HuBERT, WavLM, DistilHuBERT), providing rich and stable training signals that capture both acoustic and phonetic structure.
- **Unpaired Learning** — Natively supports training on fully unpaired noisy/clean speech data, enabling cross-dataset and cross-gender generalization without paired supervision.
- **State-of-the-Art Generalization** — Achieves state-of-the-art WV-MOS and SCOREQ on the DNS Challenge 2020 blind test set, outperforming multi-step diffusion and consistency-based baselines.

---

## 🤗 Try it Out

You can easily try out DriftSE without any local installation! Visit our **[Hugging Face Space](https://huggingface.co/spaces/LIANGXU123/DriftSE)** to upload your own noisy audio files and test the speech enhancement interactively.

---

## 📦 Repository Contents

The [Hugging Face repository](https://huggingface.co/LIANGXU123/DriftSE) hosts pre-trained checkpoints, SSL latent encoders, and enhanced audio outputs for **DriftSE**:

```
LIANGXU123/DriftSE/
├── logs/                                              # Pre-trained DriftSE checkpoints
│   ├── distillhubert_three_layers_with_z/
│   │   └── last.ckpt                                  # DriftSE (DistilHuBERT) — conditional generator
│   ├── distillhubert_three_layers_pesq_sisdr_ccmse_with_z/
│   │   └── last.ckpt                                  # DriftSE† (DistilHuBERT) — with auxiliary losses
│   ├── hubert_three_layers_with_z/
│   │   └── last.ckpt                                  # DriftSE (HuBERT) — conditional generator
│   └── wavlm_three_layers_with_z/
│       └── last.ckpt                                  # DriftSE (WavLM) — conditional generator
├── latent_ckpt/                                       # Frozen SSL speech encoders (for training/features)
│   ├── distilhubert-local/                            # DistilHuBERT (768-d, 2 layers)
│   ├── hubert-large-local/                            # HuBERT-Large (1024-d, 24 layers)
│   └── wavlm-large-local/                             # WavLM-Large (1024-d, 24 layers)
└── out.zip                                            # Pre-computed enhanced audio outputs (VB-DMD test set)
```

---

## 📊 Performance Benchmark

### VoiceBank-DEMAND (VB-DMD) — In-Domain Evaluation

| Method | NFE | PESQ (↑) | SI-SDR (↑) | ESTOI (↑) | DNSMOS (↑) | SCOREQ (↑) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| MetricGAN+ | 1 | 3.13 | 8.50 | 0.83 | 3.22 | 3.82 |
| UNIVERSE++ | 8 | 2.91 | 18.00 | 0.85 | 3.45 | **4.35** |
| SGMSE+ | 30 | 2.90 | 16.90 | 0.85 | 3.48 | 3.98 |
| ROSE-CD | 1 | 3.49 | 17.80 | 0.87 | 3.49 | 4.23 |
| SBCTM | 1 | **3.56** | 12.70 | 0.87 | 3.55 | **4.35** |
| MeanFlowSE | 1 | 2.81 | **19.97** | **0.88** | **3.58** | 4.25 |
| *DriftSE (WavLM)* | 1 | 3.03 | 14.00 | 0.85 | **3.54** | **4.17** |
| *DriftSE (HuBERT)* | 1 | 2.94 | 12.50 | 0.84 | 3.49 | 4.14 |
| *DriftSE (DistilHuBERT)* | 1 | 3.00 | 15.60 | 0.85 | 3.48 | 4.15 |
| ***DriftSE† (DistilHuBERT)*** | **1** | **3.45** | **20.60** | **0.87** | 3.49 | 4.11 |

> † Jointly trained with auxiliary PESQ, SI-SDR, and CCMSE losses.

### DNS Challenge 2020 Blind Test Set — Real-World Generalization

| Method | NFE | WV-MOS (↑) | SCOREQ (↑) | SIG (↑) | BAK (↑) | OVRL (↑) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| MetricGAN+ | 1 | 1.23 | 2.08 | 3.28 | 3.45 | 2.70 |
| UNIVERSE++ | 8 | 1.99 | 2.27 | 3.45 | 3.52 | 2.93 |
| SGMSE+ | 30 | 2.34 | **2.95** | **4.12** | **3.94** | **3.62** |
| ROSE-CD | 1 | **2.37** | 2.81 | 4.01 | 3.80 | 3.42 |
| SBCTM | 1 | 2.24 | 2.78 | 3.83 | 3.88 | 3.33 |
| MeanFlowSE | 1 | 2.20 | 2.79 | 3.88 | 3.51 | 3.21 |
| *DriftSE (WavLM)* | 1 | 2.62 | 2.67 | 3.85 | **3.94** | **3.42** |
| *DriftSE (HuBERT)* | 1 | 2.56 | 2.74 | **3.92** | 3.79 | 3.40 |
| ***DriftSE (DistilHuBERT)†*** | **1** | **2.65** | **2.97** | 3.78 | 3.84 | 3.31 |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/liangxu123/driftse.git
cd driftse
pip install -r requirements.txt
```

### 2. Download Checkpoints & Assets

Download all checkpoints and assets directly into the project directory:

```bash
huggingface-cli download LIANGXU123/DriftSE --local-dir .

# (Optional) Extract pre-generated enhanced audio outputs
unzip -q out.zip
```

Or via Python:

```python
from huggingface_hub import snapshot_download

snapshot_download("LIANGXU123/DriftSE", local_dir=".")
```

### 3. Run Enhancement

```bash
bash ./test.sh <GPU_ID> [CONFIG_PATH]

# Example: default config (DistilHuBERT, conditional generator)
bash ./test.sh 0

# Example: specific config
bash ./test.sh 0 ./config/with_z/v2_drift2_distillhubert_three_layers.json
```

The evaluation pipeline runs two phases:
1. **Enhancement** — generates enhanced audio via `enhancement.py`
2. **Objective Metrics** — computes PESQ, ESTOI, SI-SDR via `calc_metrics.py`

---

## 🏗️ Model Architecture

| Component | Details |
|---|---|
| **Backbone** | NCSN++V2 (without time embedding) |
| **Input** | Complex STFT spectrogram (510-pt Hann window, hop 128) |
| **Audio** | 16 kHz mono |
| **SSL Encoder** | Frozen DistilHuBERT / HuBERT-Large / WavLM-Large |
| **Drifting Kernel** | Multi-temperature exponential kernel (τ ∈ {0.1, 0.5, 1.0}) |
| **Inference** | Single-step (1 NFE) — no iterative denoising |
| **Optimizer** | SOAP / AdamW, lr = 5×10⁻⁴, weight decay = 0.01 |
| **Training** | 100 epochs, batch size 14 × 4 gradient accumulation |

### Two Formulations

- **Conditional Generator (`with_z/`)** — Stochastic mapping `f_θ(ε, y)` from Gaussian noise conditioned on noisy speech, optimized for perceptual quality (DNSMOS, SCOREQ).
- **Direct Mapping (`no_z/`)** — Deterministic mapping `f_θ(y)` from noisy to clean speech, with `σ=0` for highest PESQ/SI-SDR fidelity.

---

## 🤗 SSL Encoder Checkpoints

DriftSE uses frozen self-supervised speech encoders to compute the latent drifting field during training (already included if you downloaded the repository via step 2; **not** needed for inference).

To download only the latent SSL encoders:

```bash
huggingface-cli download LIANGXU123/DriftSE --include "latent_ckpt/*" --local-dir .
```

Or via Python:

```python
from huggingface_hub import snapshot_download

snapshot_download("LIANGXU123/DriftSE", allow_patterns=["latent_ckpt/*"], local_dir=".")
```

```
latent_ckpt/
├── wavlm-large-local/       # WavLM-Large (1024-d, 24 layers)
├── hubert-large-local/      # HuBERT-Large (1024-d, 24 layers)
└── distilhubert-local/      # DistilHuBERT (768-d, 2 layers)
```

---

## 🔧 Training

To train DriftSE from scratch:

```bash
bash ./train.sh <GPU_ID> [CONFIG_PATH]

# Example: default DistilHuBERT config
bash ./train.sh 0

# Example: with auxiliary losses (PESQ + SI-SDR + CCMSE)
bash ./train.sh 0 ./config/with_z/v2_drift2_distillhubert_three_layers_pesq_sisdr_ccmse.json
```

Training uses **dynamic mixing**: 10,802 clean VoiceBank utterances are mixed on-the-fly with 18 DEMAND noise types at SNRs sampled from {0, 5, 10, 15} dB.

---

## 📝 Citation

If you find DriftSE useful in your research, please cite:

```bibtex
@inproceedings{xu2026driftse,
  author    = {Liang Xu and Diego Caviedes-Nozal and W. Bastiaan Kleijn and Longfei Felix Yan and Rasmus Kongsgaard Olsson},
  title     = {Speech Enhancement Based on Drifting Models},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026}
}
```
```bibtex
@article{xu2026driftsespeechenhancementgenerative,
  author  = {Xu, Liang and Caviedes-Nozal, Diego and Kleijn, W. Bastiaan and Yan, Longfei Felix and Olsson, Rasmus Kongsgaard},
  title   = {Speech Enhancement Based on Drifting Models},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2026},
  note    = {Submitted},
}
```

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙏 Acknowledgments

We thank the authors of [SGMSE+](https://github.com/sp-uhh/sgmse) and [Drifting Models](https://github.com/tyfeld/drifting-model) for the foundational work that inspired this codebase.
