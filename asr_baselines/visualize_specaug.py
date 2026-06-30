"""Standalone SpecAugment sanity experiment.

Renders before/after log-mel spectrograms so the Green-style masking can be
eyeballed, and prints the actual masked fraction (freq vs time) over N draws so
the config is validated independently of any model. Works on a real patient wav
(--wav / --root-dir+--label-file) or a synthetic sweep if no audio is given.

  python -m asr_baselines.visualize_specaug --config asr_baselines/configs/specaug_green.yaml
  python -m asr_baselines.visualize_specaug --wav clip.wav --out specaug.png
"""

import argparse
import os

import torch
import torchaudio
import yaml

from .specaugment import SpecAugmentConfig, SpecAugment

N_MELS = 80
SR = 16000


def make_logmel(wav: torch.Tensor) -> torch.Tensor:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=400, hop_length=160, n_mels=N_MELS
    )(wav)
    logmel = torch.log(mel + 1e-6)
    # mean/std normalize so the 0-fill of masking sits at the mean.
    return (logmel - logmel.mean()) / (logmel.std() + 1e-6)


def synthetic_logmel(seconds=6.0) -> torch.Tensor:
    t = torch.linspace(0, seconds, int(seconds * SR))
    wav = sum(torch.sin(2 * torch.pi * f * t) for f in (200, 600, 1500, 3000)) / 4
    return make_logmel(wav)


def masked_fraction_report(aug: SpecAugment, mel: torch.Tensor, draws=50):
    aug.train()
    base_zero = (mel == 0).float().mean().item()
    fracs = []
    for _ in range(draws):
        m = aug(mel.clone())
        fracs.append((m == 0).float().mean().item())
    fracs = torch.tensor(fracs)
    print(f"baseline zero-fraction: {base_zero:.3f}")
    print(f"masked zero-fraction over {draws} draws: "
          f"mean={fracs.mean():.3f} min={fracs.min():.3f} max={fracs.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--wav", default=None)
    ap.add_argument("--root-dir", default=None)
    ap.add_argument("--label-file", default=None)
    ap.add_argument("--sp-model-path",
                    default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         "spm", "spm_unigram_1023.model"))
    ap.add_argument("--out", default="specaug.png")
    args = ap.parse_args()

    cfg_dict = None
    if args.config:
        with open(args.config) as f:
            cfg_dict = (yaml.safe_load(f) or {}).get("specaugment")
    aug = SpecAugment(SpecAugmentConfig.from_dict(cfg_dict))
    print("config:", aug.config)

    if args.wav:
        wav, sr = torchaudio.load(args.wav)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        mel = make_logmel(wav.mean(0))
    elif args.root_dir and args.label_file:
        from .patient_audio import load_audio_examples, load_waveform
        ex = load_audio_examples(args.root_dir, args.label_file, args.sp_model_path, limit=1)[0]
        print("sample text:", ex.text)
        mel = make_logmel(load_waveform(ex.path))
    else:
        mel = synthetic_logmel()

    masked_fraction_report(aug, mel)

    aug.train()
    after = aug(mel.clone())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].imshow(mel.numpy(), origin="lower", aspect="auto")
        axes[0].set_title("log-mel (clean)")
        axes[1].imshow(after.numpy(), origin="lower", aspect="auto")
        axes[1].set_title(f"log-mel + Green SpecAugment ({aug.config})")
        for a in axes:
            a.set_ylabel("mel bin")
        axes[1].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(args.out, dpi=110)
        print(f"wrote {args.out}")
    except ImportError:
        print("matplotlib not available; skipped figure")


if __name__ == "__main__":
    main()
