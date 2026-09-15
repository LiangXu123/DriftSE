from .config_loader import load_config, _deep_merge

def __getattr__(name):
    if name == "SpeechDataset":
        from .speech_dataset import SpeechDataset
        return SpeechDataset
    elif name in ("compute_V", "compute_V_paired"):
        from . import drifting
        return getattr(drifting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
