# Speech Enhancement Based on Drifting Models (DriftSE)

This work introduces **DriftSE**, a new generative framework for speech enhancement that reformulates denoising as a **distribution-level evolution problem**, rather than traditional paired regression or iterative diffusion sampling.

---

## Core Idea

Instead of learning a direct mapping from noisy speech to clean speech, or relying on multi-step iterative refinement, DriftSE models speech enhancement as the problem of **aligning probability distributions**.

The method learns a **drifting field**, a vector field that continuously transports samples from the noisy speech distribution toward regions of higher probability in the clean speech distribution.

In this formulation, speech enhancement is not a sequence of denoising steps, but the result of an **equilibrium process** where the source distribution is gradually transformed into the target distribution.

---

## Main Contribution

The main contribution is a **drift-based generative formulation of speech enhancement**, where:

- Speech enhancement is defined as an **equilibrium distribution matching problem**
- A learned **drifting field** guides noisy samples toward clean speech regions
- The model enables **one-step inference**, removing the need for iterative sampling

This fundamentally differs from diffusion models, which require multiple denoising steps at inference time.

---

## Learning Strategy

DriftSE is trained by matching distributions rather than relying strictly on paired supervision.

The framework explores two formulations:

### 1. Direct Mapping Formulation
A deterministic model learns to transform noisy speech directly into clean speech, guided by the drifting field.

### 2. Stochastic Generative Formulation
A generative model starts from a simple prior distribution (e.g., Gaussian noise) and uses the learned drift dynamics to generate clean speech samples.

---

## Intuition

The drifting field acts as a learned correction mechanism:

- Noisy or low-quality speech samples are pushed toward cleaner regions of the speech manifold
- The model learns the **geometry of clean speech distributions**
- Enhancement emerges from **distributional transport**, not explicit noise removal

---

## Experimental Findings

Experiments on the VoiceBank-DEMAND benchmark show that DriftSE:

- Produces high-quality speech enhancement in **a single inference step**
- Matches or outperforms multi-step diffusion-based baselines
- Generalizes well to unseen noise types and real-world conditions

Despite eliminating iterative inference, the model maintains strong perceptual quality and intelligibility.

---

## Key Insight

The central insight is that speech enhancement can be reframed as a **continuous transformation of probability distributions**, rather than a step-by-step denoising process.

By learning a drifting field that governs this transformation, DriftSE provides a **fast, non-iterative alternative to diffusion-based speech enhancement**, while preserving high-quality reconstruction performance.
