import hashlib
import os
import urllib.parse
import urllib.request
from typing import Dict, Iterable, Tuple


def preflight_environment() -> Dict[str, str]:
    results = {}
    import torch
    import torchaudio
    import sentencepiece
    import pytorch_lightning
    from torchaudio.models import RNNTBeamSearch

    try:
        from torchaudio.models.rnnt import emformer_rnnt_model
    except ImportError:
        from torchaudio.models import emformer_rnnt_model

    if not hasattr(RNNTBeamSearch, "infer"):
        raise RuntimeError("torchaudio.models.RNNTBeamSearch.infer is required for streaming")
    if not hasattr(torchaudio.transforms, "RNNTLoss"):
        raise RuntimeError("torchaudio.transforms.RNNTLoss is required for online AVSR training")
    results["torch"] = torch.__version__
    results["torchaudio"] = torchaudio.__version__
    results["sentencepiece"] = getattr(sentencepiece, "__version__", "unknown")
    results["pytorch_lightning"] = pytorch_lightning.__version__
    results["emformer_rnnt_model"] = emformer_rnnt_model.__name__
    results["rnnt_infer"] = "available"
    return results


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(url: str, dest_dir: str, expected_sha256: str = "") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(urllib.parse.urlparse(url).path) or "online_avsr.ckpt"
    dest = os.path.join(dest_dir, name)
    if not os.path.isfile(dest):
        urllib.request.urlretrieve(url, dest)
    if expected_sha256:
        actual = sha256_file(dest)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(f"SHA256 mismatch for {dest}: expected {expected_sha256}, got {actual}")
    return dest


def extract_state_dict(checkpoint) -> Tuple[dict, bool]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary")
    if isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"], True
    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint, False
    raise ValueError("Checkpoint does not contain a usable state_dict")


def _has_any_prefix(keys: Iterable[str], prefixes) -> bool:
    return any(any(k.startswith(prefix) for prefix in prefixes) for k in keys)


def validate_online_state_dict(state_dict: dict) -> None:
    keys = list(state_dict.keys())
    rnnt_key = any(
        k.startswith("model.") and any(token in k for token in ("transcriber", "predictor", "joiner"))
        for k in keys
    )
    # At least one frontend must be present (audiovisual has both; audio-only
    # or video-only checkpoints have just one and no fusion).
    has_frontend = _has_any_prefix(keys, ["audio_frontend.", "video_frontend."])
    if not rnnt_key or not has_frontend:
        raise ValueError(
            "Checkpoint does not look like an online AVSR RNN-T checkpoint; "
            f"rnnt_key_found={rnnt_key}, frontend_found={has_frontend}"
        )
    offline_markers = ("encoder.", "aux_encoder.", "decoder.", "ctc.")
    if any(k.startswith(offline_markers) for k in keys):
        raise ValueError("Checkpoint looks like the current offline Auto-AVSR Conformer checkpoint")


def load_validated_state_dict(path: str, map_location="cpu") -> dict:
    import torch

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict, _ = extract_state_dict(checkpoint)
    validate_online_state_dict(state_dict)
    return state_dict


def load_online_avsr_module(checkpoint_path: str, sp_model_path: str, device):
    from .module import OnlineAVSRModule
    from .text import load_sentencepiece_model

    sp_model = load_sentencepiece_model(sp_model_path)
    module = OnlineAVSRModule(sp_model=sp_model)
    state_dict = load_validated_state_dict(checkpoint_path, map_location=device)
    module.load_state_dict(state_dict, strict=True)
    module.to(device)
    module.eval()
    return module
