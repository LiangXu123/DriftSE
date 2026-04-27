## Speech Enhancement Based on Drifting Models (DriftSE)

🔗 Project page: https://liangxu123.github.io/driftse/  
🔗 Paper: [DriftSE](https://www.researchgate.net/publication/404224466_Speech_Enhancement_Based_on_Drifting_Models  )

---

**DriftSE** is a speech enhancement method that reformulates denoising as a **distribution transport problem**.

Instead of iterative denoising or direct regression, it learns a **drifting field** that moves noisy speech distributions toward clean speech.

---

### Key idea
- Speech enhancement = **distribution evolution**
- Learn a **drift field** to guide noisy → clean speech
- Enables **one-step inference**

---

### Summary
DriftSE replaces iterative denoising with a learned continuous transformation of distributions, achieving fast and effective speech enhancement in a single step.
