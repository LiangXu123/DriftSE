# DriftSE: Speech Enhancement with Generative Drifting

> **DriftSE: Speech Enhancement with Generative Drifting** (Submitted to IEEE/ACM TASLP for possible publication. )
>
> *Liang Xu, Diego Caviedes-Nozal, W. Bastiaan Kleijn, Longfei Felix Yan, Rasmus Kongsgaard Olsson*

🔗 [**Project Website**](https://liangxu123.github.io/) &nbsp;|&nbsp; 📄 [**arXiv Paper**]([https://arxiv.org/abs/2604.24199](https://arxiv.org/abs/2609.12252)) &nbsp;|&nbsp; 🤗 [**Hugging Face Space**](https://huggingface.co/spaces/LIANGXU123/DriftSE)

---

## ✨ Key Highlights

- **Dual Latent Drifting** — Leverages dual-branch latent representations, combining speech semantic encoders (WavLM, HuBERT, DistilHuBERT) with acoustic encoders (BEATs and PANNs) and WavCube-pro for joint semantic-acoustic latents to provide rich, complementary training signals capturing phonetic, acoustic, and structural details.
- **Unpaired Learning** — Natively supports training on fully unpaired noisy/clean speech data, enabling cross-dataset and cross-gender generalization without paired supervision.
- **Novel Generative Paradigm** — Formulates speech enhancement as a distributional equilibrium problem, eliminating the need for iterative denoising or trajectory-based sampling.
- **Native One-Step Inference** — Achieves single-step (1 NFE) enhancement by evolving the pushforward distribution of a mapping function to directly match the clean speech distribution via a Drifting Field.
- **State-of-the-Art Generalization** — Achieves state-of-the-art performance on four test sets, outperforming multi-step diffusion and other baselines.

---

## 🤗 Try it Out

You can easily try out DriftSE without any local installation! Visit our **[Hugging Face Space](https://huggingface.co/spaces/LIANGXU123/DriftSE)** to upload your own noisy audio files and test the speech enhancement interactively.

---

## 📦 Repository Contents

The [Hugging Face repository](https://huggingface.co/LIANGXU123/DriftSE/tree/dual-latent-DriftSE) hosts pre-trained checkpoints, SSL latent encoders, and enhanced audio outputs for **DriftSE**:

```
LIANGXU123/DriftSE/
├── logs/                                              # Pre-trained DriftSE checkpoints
│   ├── xxxx/
│   │   └── last.ckpt                             
├── latent_ckpt/                                       # Frozen SSL speech encoders (for training/features)
│   ├── distilhubert-local/                            # DistilHuBERT (768-d, 2 layers)
│   ├── hubert-large-local/                            # HuBERT-Large (1024-d, 24 layers)
│   ├── wavlm-large-local/                             # WavLM-Large (1024-d, 24 layers)
│   ├── beats-local/                                   # BEATs (768-d, 12 layers)
│   ├── panns-local/                                   # PANNs CNN14 (1024-d / 2048-d)
│   ├── WavCube/                                       # WavCube-pro (128-d, joint semantic-acoustic)
└── out.zip                                            # Pre-computed enhanced audio outputs
```

---

## 📊 Performance Benchmark

### EARS-WHAM — Speech Denoising

| Model                    | Causal |       Latent       | Params |     WER (↓)     |   PESQ (↑)   |   SI-SDR (↑)   |   ESTOI (↑)   |  SCOREQ (↑)  |
| ------------------------ | :----: | :----------------: | :----: | :--------------: | :------------: | :-------------: | :------------: | :------------: |
| *Noisy*                  |   -    |         -          |   -    |      32.80%      |      1.24      |       5.4       |      0.64      |      2.13      |
| **Non-Causal**           |        |                    |        |                  |                |                |                |                |
| SGMSE+                   |   ❌   |         -          |  65M   |      18.65%      |      2.20      |      14.2      |      0.84      |      3.48      |
| ROSE-CD                  |   ❌   |         -          | 59.62M |    **18.19%**    |    **2.81**    |      15.3      |      0.85      |    **3.50**    |
| DM-IERM                  |   ❌   |         -          |  67M   |        -         |      2.67      |    **17.4**    |      0.74      |       -        |
| FM-Euler4                |   ❌   |         -          | 73.7M  |      18.40%      |      2.41      |      16.1      |    **0.86**    |       -        |
| *DriftSE (NCSN++)*       |   ❌   |       PANNs        | 59.62M |      24.34%      |      2.09      |       0.8       |      0.79      |      2.72      |
| *DriftSE (NCSN++)*       |   ❌   |    DistilHuBERT    | 59.62M |      15.19%      |      2.39      |      12.1      |      0.84      |      3.82      |
| *DriftSE (NCSN++)*       |   ❌   | DistilHuBERT+PANNs | 59.62M |    **14.33%**    |      2.46      |    **13.5**    |    **0.85**    |    **3.85**    |
| *DriftSE (NCSN++)*       |   ❌   |      WavCube       | 59.62M |      15.35%      |      2.42      |      12.0      |      0.84      |    **3.85**    |
| *DriftSE (TF-GridNet)*   |   ❌   |      WavCube       | 1.69M  |      15.48%      |    **2.48**    |      11.2      |      0.84      |    **3.85**    |
| **Causal**               |        |                    |        |                  |                |                |                |                |
| SFM-LRK4                 |   ✔️   |         -          | 52.5M  |      20.10%      |      2.30      |      14.1      |      0.83      |       -        |
| *DriftSE (SFMUnet)*      |   ✔️   |       PANNs        | 24.6M  |      27.15%      |      1.82      |      -41.6      |      0.10      |      2.58      |
| *DriftSE (SFMUnet)*      |   ✔️   |    DistilHuBERT    | 24.6M  |      19.21%      |      2.05      |    **-31.5**   |      0.51      |      3.33      |
| *DriftSE (SFMUnet)*      |   ✔️   | DistilHuBERT+PANNs | 24.6M  |      18.67%      |      2.13      |      -31.6      |      0.52      |      3.30      |
| *DriftSE (SFMUnet)*      |   ✔️   |      WavCube       | 24.6M  |      18.94%      |      2.21      |      -33.9      |      0.51      |      3.35      |
| *DriftSE (TF-GridNet)*   |   ✔️   |      WavCube       | 1.24M  |    **18.11%**    |    **2.26**    |      -33.8      |    **0.53**    |    **3.47**    |

### EARS-REVERB — Speech Dereverberation

| Model                    | Causal |       Latent       | Params |     WER (↓)     |   PESQ (↑)   |   SI-SDR (↑)   |   ESTOI (↑)   |  SCOREQ (↑)  |
| ------------------------ | :----: | :----------------: | :----: | :--------------: | :------------: | :-------------: | :------------: | :------------: |
| *Reverberant*            |   -    |         -          |   -    |      20.10%      |      1.32      |      -16.6      |      0.58      |      3.12      |
| **Non-Causal**           |        |                    |        |                  |                |                |                |                |
| SGMSE+                   |   ❌   |         -          | 65.59M |      17.32%      |      1.95      |      -12.6      |      0.76      |    **2.69**    |
| ROSE-CD                  |   ❌   |         -          | 59.62M |      15.78%      |      2.69      |      -13.9      |      0.82      |      2.66      |
| DM-IERM                  |   ❌   |         -          |  67M   |        -         |    **3.52**    |    **14.2**    |    **0.92**    |       -        |
| FM-Euler5                |   ❌   |         -          | 38.7M  |    **11.40%**    |      2.31      |      -11.7      |      0.85      |       -        |
| *DriftSE (NCSN++)*       |   ❌   |    WavLM+PANNs     | 59.62M |    **8.91%**     |      2.35      |      -9.3       |      0.83      |    **3.61**    |
| *DriftSE (NCSN++)*       |   ❌   |      WavCube       | 59.62M |      10.28%      |      2.33      |      -10.5      |      0.82      |      3.30      |
| *DriftSE (TF-GridNet)*   |   ❌   |    WavLM+PANNs     | 1.69M  |      15.59%      |      2.07      |      -8.2       |      0.78      |      3.34      |
| *DriftSE (TF-GridNet)*   |   ❌   |      WavCube       | 1.69M  |      9.93%       |    **2.43**    |    **-6.8**    |    **0.84**    |      3.17      |
| **Causal**               |        |                    |        |                  |                |                |                |                |
| SFM-LRK5                 |   ✔️   |         -          | 27.9M  |      15.90%      |      2.05      |      -13.5      |      0.79      |       -        |
| *DriftSE (SFMUnet)*      |   ✔️   |    WavLM+PANNs     | 24.6M  |      10.97%      |      2.11      |    **-33.3**   |    **0.58**    |      3.28      |
| *DriftSE (SFMUnet)*      |   ✔️   |      WavCube       | 24.6M  |      11.17%      |      2.32      |      -34.3      |      0.54      |      3.30      |
| *DriftSE (TF-GridNet)*   |   ✔️   |    WavLM+PANNs     | 1.24M  |      9.00%       |      2.33      |      -33.8      |      0.56      |      3.33      |
| *DriftSE (TF-GridNet)*   |   ✔️   |      WavCube       | 1.24M  |    **8.85%**     |    **2.71**    |      -34.5      |      0.56      |    **3.58**    |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/liangxu123/driftse.git
cd driftse
git checkout dual-latent-DriftSE
pip install -r requirements.txt
```

### 2. Download Checkpoints & Assets

Download all checkpoints and assets directly into the project directory:

```bash
huggingface-cli download LIANGXU123/DriftSE --revision dual-latent-DriftSE --local-dir .

# (Optional) Extract pre-generated enhanced audio outputs
unzip -q out.zip
```

Or via Python:

```python
from huggingface_hub import snapshot_download

snapshot_download("LIANGXU123/DriftSE", revision="dual-latent-DriftSE", local_dir=".")
```

### 3. Run Enhancement

```bash
bash ./test.sh <GPU_ID> [CONFIG_PATH]

# Example: specific config
bash ./test.sh 0 ./config/DriftSE/experiments/SE_EARS_distilhubert_ch1282_incond_PANNs.json
```

The evaluation pipeline runs two phases:

1. **Enhancement** — generates enhanced audio via `enhancement.py`
2. **Objective Metrics** — computes PESQ, ESTOI, SI-SDR via `calc_metrics.py`

---

## 🔧 Training

To train DriftSE from scratch:

```bash
bash ./train.sh <GPU_ID> [CONFIG_PATH]

# Example: specific config
bash ./train.sh 0 ./config/DriftSE/experiments/SE_EARS_distilhubert_ch1282_incond_PANNs.json
```

---

## 🤗 SSL & Latent Encoder Checkpoints

DriftSE uses frozen self-supervised speech encoders and auxiliary feature extractors to compute the latent drifting field during training (already included if you downloaded the repository via step 2; **not** needed for inference).

To download only the latent encoders:

```bash
huggingface-cli download LIANGXU123/DriftSE --revision dual-latent-DriftSE --include "latent_ckpt/*" --local-dir .
```

Or via Python:

```python
from huggingface_hub import snapshot_download

snapshot_download("LIANGXU123/DriftSE", revision="dual-latent-DriftSE", allow_patterns=["latent_ckpt/*"], local_dir=".")
```

```
latent_ckpt/
├── distilhubert-local/                            # DistilHuBERT (768-d, 2 layers)
├── hubert-large-local/                            # HuBERT-Large (1024-d, 24 layers)
├── wavlm-large-local/                             # WavLM-Large (1024-d, 24 layers)
├── beats-local/                                   # BEATs (768-d, 12 layers)
├── panns-local/                                   # PANNs CNN14 (1024-d / 2048-d)
├── WavCube/                                       # WavCube-pro (128-d, joint semantic-acoustic)
```

### Verification & Testing

You can verify and test all feature encoders (WavLM, HuBERT, DistilHuBERT, BEATs, PANNs CNN14, WavCube-pro) to ensure their checkpoints load properly and inspect output tensor shapes:

```bash
python test_encoder.py --gpu 0
```

> **Note on [WavCube-pro](https://huggingface.co/yhaha/WavCube/tree/main/WavCube-pro) Customization**: We added a custom `inf_new()` method to `WavLMVAEFeatures` (`latent_ckpt/WavCube/vocos/feature_extractors.py`). This allows directly passing GPU waveform tensors (bypassing CPU processor re-processing) to compute the projected 128-d joint semantic-acoustic latent representation $z$.

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
