import torch

def resolve_torch_device() -> torch.device:
    """Return best available torch device."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")
