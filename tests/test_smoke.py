"""CPU smoke tests for the streaming Emformer RNN-T port.

Run: .venv/bin/python -m tests.test_smoke
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_avsr.module import AVBatch, OnlineAVSRModule  # noqa: E402
from online_avsr.streaming import (  # noqa: E402
    EagerBackend,
    RoiPreprocessor,
    StreamingInferencePipeline,
    iter_av_windows,
)


class FakeSpm:
    """SentencePiece stand-in with the expected 1023-piece vocabulary."""

    def get_piece_size(self):
        return 1023

    def unk_id(self):
        return 2

    def eos_id(self):
        return 1

    def pad_id(self):
        return 0

    def id_to_piece(self, ids):
        return [f"▁t{i}" for i in ids]

    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def make_module(segment_length=8, right_context_length=0):
    torch.manual_seed(0)
    args = SimpleNamespace(
        segment_length=segment_length,
        right_context_length=right_context_length,
        learning_rate=8e-4,
        epochs=2,
    )
    module = OnlineAVSRModule(args=args, sp_model=FakeSpm())
    module.log = lambda *a, **k: None  # no trainer attached
    module.eval()
    return module


def test_training_step_loss():
    module = make_module()
    frames = 16
    batch = AVBatch(
        audios=torch.randn(1, frames * 640, 1),
        videos=torch.randn(1, frames, 1, 88, 88),
        audio_lengths=torch.tensor([frames], dtype=torch.int32),
        video_lengths=torch.tensor([frames], dtype=torch.int32),
        targets=torch.randint(3, 1023, (1, 5), dtype=torch.int32),
        target_lengths=torch.tensor([5], dtype=torch.int32),
    )
    module.train()
    loss = module._step(batch, "train")
    assert torch.isfinite(loss), f"non-finite RNNT loss: {loss}"
    print(f"ok: training step loss = {loss.item():.2f}")


def test_encode_chunk_trims_context():
    module = make_module()
    frames, ctx = 12, 4
    with torch.no_grad():
        fused = module.encode_chunk(
            torch.randn(1, frames * 640, 1), torch.randn(1, frames, 1, 88, 88), context_frames=ctx
        )
    assert fused.shape[1] == frames - ctx == 8, f"expected 8 fused frames, got {fused.shape}"
    assert fused.shape[2] == 512
    print(f"ok: encode_chunk -> {tuple(fused.shape)}")


def test_streaming_state_carry():
    module = make_module(segment_length=8)
    pipeline = StreamingInferencePipeline(
        EagerBackend(module), RoiPreprocessor(), FakeSpm(), beam_width=5, carry_state=True
    )
    total_frames = 20  # 3 windows of 8 (last padded)
    video = np.random.randint(0, 255, (total_frames, 96, 96, 3), dtype=np.uint8)
    audio = torch.randn(total_frames * 640, 1)

    n_chunks = 0
    for start, end, v_win, a_win in iter_av_windows(video, audio, step_frames=8, lookback_frames=4):
        transcript, new, n_feats = pipeline.infer_chunk(v_win, a_win, trim_frames=4)
        assert n_feats == 8, f"Emformer.infer needs exactly segment frames, got {n_feats}"
        n_chunks += 1
    assert n_chunks == 3
    assert pipeline.state is not None, "decoder state was not carried"
    print(f"ok: 3 streaming chunks, state carried, transcript={transcript!r}")


if __name__ == "__main__":
    test_training_step_loss()
    test_encode_chunk_trims_context()
    test_streaming_state_carry()
    print("all smoke tests passed")
