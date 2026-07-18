"""
Drifting Models Implementation

A PyTorch reproduction of "Generative Modeling via Drifting" (Deng et al., 2026)
for MNIST and CIFAR-10 datasets.
"""

from .utils import (
    EMA,
    WarmupLRScheduler,
    SampleQueue,
    save_checkpoint,
    load_checkpoint,
    save_image_grid,
    visualize_samples,
    count_parameters,
    set_seed,
)

__version__ = "0.1.0"
__all__ = [
    # Utils
    "EMA",
    "WarmupLRScheduler",
    "SampleQueue",
    "save_checkpoint",
    "load_checkpoint",
    "save_image_grid",
    "visualize_samples",
    "count_parameters",
    "set_seed",
]
