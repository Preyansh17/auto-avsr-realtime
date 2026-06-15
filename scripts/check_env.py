#!/usr/bin/env python3
"""Check whether the active environment can run the streaming AVSR fork.

Run it inside whatever conda env you intend to use (e.g. auto_avsr) and it
reports, per dependency: version, whether it imports, and whether the
specific APIs the fork needs are present. Ends with a verdict for the two
use cases (inference/demo/eval vs training), so an env that's only missing
the training bits is still usable for inference.

  python scripts/check_env.py
"""

import importlib
import sys

# (module, pip name, min_version_major_for_warn, tags)
CHECKS = [
    ("torch", "torch", None, "both"),
    ("torchaudio", "torchaudio", None, "both"),
    ("torchvision", "torchvision", None, "both"),
    ("pytorch_lightning", "pytorch-lightning", 2, "training"),
    ("sentencepiece", "sentencepiece", None, "both"),
    ("mediapipe", "mediapipe", None, "inference"),
    ("cv2", "opencv-python", None, "inference"),
    ("skimage", "scikit-image", None, "inference"),
    ("av", "av", None, "inference"),
    ("soundfile", "soundfile", None, "both"),
    ("numpy", "numpy", None, "both"),
]


def version_of(mod):
    for attr in ("__version__", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "?"


def main():
    print(f"python {sys.version.split()[0]}\n")
    missing_inference, missing_training, warnings = [], [], []

    for name, pip_name, _, tag in CHECKS:
        try:
            mod = importlib.import_module(name)
            print(f"  OK    {pip_name:18s} {version_of(mod)}")
        except Exception as exc:
            print(f"  FAIL  {pip_name:18s} (import error: {exc})")
            (missing_training if tag == "training" else missing_inference).append(pip_name)
            if tag == "both":
                missing_training.append(pip_name)
            continue

    print()

    # --- targeted API checks --------------------------------------------------
    # torchaudio RNN-T streaming APIs (needed everywhere)
    try:
        import torchaudio
        from torchaudio.models import RNNTBeamSearch  # noqa: F401

        try:
            from torchaudio.models.rnnt import emformer_rnnt_model  # noqa: F401
        except ImportError:
            from torchaudio.models import emformer_rnnt_model  # noqa: F401
        assert hasattr(RNNTBeamSearch, "infer"), "RNNTBeamSearch.infer missing"
        assert hasattr(torchaudio.transforms, "RNNTLoss"), "RNNTLoss missing"
        print("  OK    torchaudio RNN-T APIs (RNNTBeamSearch.infer, RNNTLoss, emformer_rnnt_model)")
    except Exception as exc:
        print(f"  FAIL  torchaudio RNN-T APIs: {exc}")
        missing_inference.append("torchaudio RNN-T APIs")
        missing_training.append("torchaudio RNN-T APIs")

    # mediapipe legacy solutions API (vendored face/mouth detectors need it)
    try:
        import mediapipe as mp

        assert hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")
        print("  OK    mediapipe legacy mp.solutions.face_detection")
    except Exception as exc:
        print(f"  WARN  mediapipe mp.solutions unavailable: {exc}")
        warnings.append("mediapipe lacks mp.solutions (need <=0.10.21) -- "
                        "demo --preprocess face/mouth will fail; --preprocess roi still works")

    # pytorch-lightning major version (train.py uses the 2.x Trainer API)
    try:
        import pytorch_lightning as pl

        major = int(pl.__version__.split(".")[0])
        if major < 2:
            warnings.append(f"pytorch-lightning {pl.__version__} is <2.x; train.py needs 2.x "
                            "(pip install -U 'pytorch-lightning>=2.4,<3')")
            print(f"  WARN  pytorch-lightning {pl.__version__} (<2.x; training needs 2.x)")
        else:
            print(f"  OK    pytorch-lightning {pl.__version__} (>=2.x)")
    except Exception:
        pass

    # --- verdict --------------------------------------------------------------
    print("\n" + "=" * 60)
    if not missing_inference:
        print("READY for inference (demo_realtime.py, eval.py, bootstrap).")
    else:
        print(f"NOT ready for inference; missing: {sorted(set(missing_inference))}")
    if not missing_training and not any("lightning" in w for w in warnings):
        print("READY for training (train.py).")
    else:
        extra = [w for w in warnings if "lightning" in w]
        print(f"NOT ready for training; fix: {sorted(set(missing_training)) + extra}")
    if warnings:
        print("\nNotes:")
        for w in warnings:
            print(f"  - {w}")
    print("=" * 60)
    return 0 if not missing_inference else 1


if __name__ == "__main__":
    sys.exit(main())
