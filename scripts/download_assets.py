#!/usr/bin/env python3
"""Download the device_avsr tutorial assets (pretrained streaming AVSR).

Fetches the TorchScript model into cpts/ and the 1023-piece SentencePiece
model into spm/, via torchaudio's asset cache.
"""

import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    from torchaudio.utils import download_asset

    jit_src = download_asset("tutorial-assets/device_avsr_model.pt")
    spm_src = download_asset("tutorial-assets/spm_unigram_1023.model")

    cpts = os.path.join(PROJECT_ROOT, "cpts")
    os.makedirs(cpts, exist_ok=True)
    jit_dst = os.path.join(cpts, "device_avsr_model.pt")
    spm_dst = os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model")
    shutil.copyfile(jit_src, jit_dst)
    shutil.copyfile(spm_src, spm_dst)
    print(f"JIT model: {jit_dst} ({os.path.getsize(jit_dst) / 1e6:.1f} MB)")
    print(f"SentencePiece model: {spm_dst} ({os.path.getsize(spm_dst) / 1e3:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
