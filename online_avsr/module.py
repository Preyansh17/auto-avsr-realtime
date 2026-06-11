import itertools
from collections import namedtuple

import torch
import torchaudio
from pytorch_lightning import LightningModule
from torchaudio.models import RNNTBeamSearch

from . import EXPECTED_SPM_VOCAB_SIZE
from .models import (
    audio_resnet,
    emformer_rnnt,
    emformer_rnnt_device,
    fusion_module,
    video_linear,
    video_resnet,
)
from .schedulers import WarmupCosineScheduler
from .text import post_process_hypotheses


AVBatch = namedtuple("AVBatch", ["audios", "videos", "audio_lengths", "video_lengths", "targets", "target_lengths"])


class OnlineAVSRModule(LightningModule):
    def __init__(self, args=None, sp_model=None):
        super().__init__()
        self.save_hyperparameters(ignore=["sp_model"])
        self.args = args
        self.sp_model = sp_model
        vocab_size = self.sp_model.get_piece_size()
        if vocab_size != EXPECTED_SPM_VOCAB_SIZE:
            raise ValueError(
                f"Expected SentencePiece vocab size {EXPECTED_SPM_VOCAB_SIZE}, got {vocab_size}"
            )
        self.blank_idx = vocab_size
        # "recipe": examples/avsr config (Conv3dResNet video frontend, 20-layer
        # Emformer, segment 64/rc 0, trained on 88x88 mouth ROIs).
        # "device": the published device_avsr small model (Linear video
        # frontend on 44x44 face crops, 12-layer Emformer, segment 32/rc 4).
        self.architecture = getattr(args, "architecture", "recipe") or "recipe"
        if self.architecture == "device":
            self.segment_length = int(getattr(args, "segment_length", 32) or 32)
            rc = getattr(args, "right_context_length", None)
            self.right_context_length = 4 if rc is None else int(rc)
            self.audio_frontend = audio_resnet()
            self.video_frontend = video_linear()
            self.fusion = fusion_module(hidden_dim=1024)
            self.model = emformer_rnnt_device(
                segment_length=self.segment_length,
                right_context_length=self.right_context_length,
            )
        else:
            self.segment_length = int(getattr(args, "segment_length", 64) or 64)
            self.right_context_length = int(getattr(args, "right_context_length", 0) or 0)
            self.audio_frontend = audio_resnet()
            self.video_frontend = video_resnet()
            self.fusion = fusion_module()
            self.model = emformer_rnnt(
                segment_length=self.segment_length,
                right_context_length=self.right_context_length,
            )
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum")
        self._decoder = None

    def configure_optimizers(self):
        args = self.args
        lr = float(getattr(args, "learning_rate", 8e-4) or 8e-4)
        optimizer = torch.optim.AdamW(
            itertools.chain(
                self.model.parameters(),
                self.video_frontend.parameters(),
                self.audio_frontend.parameters(),
                self.fusion.parameters(),
            ),
            lr=lr,
            weight_decay=0.06,
            betas=(0.9, 0.98),
        )
        warmup_epochs = int(getattr(args, "warmup_epochs", 10) or 10)
        total_epochs = int(getattr(args, "epochs", 55) or 55)
        steps_per_epoch = (
            len(self.trainer.datamodule.train_dataloader())
            / self.trainer.num_devices
            / self.trainer.num_nodes
        )
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @property
    def decoder(self) -> RNNTBeamSearch:
        if self._decoder is None:
            self._decoder = RNNTBeamSearch(self.model, self.blank_idx)
        return self._decoder

    def encode_av(self, audios, videos):
        video_features = self.video_frontend(videos)
        audio_features = self.audio_frontend(audios)
        length = min(video_features.size(1), audio_features.size(1))
        if length <= 0:
            raise ValueError("Audio/video frontends produced no frames")
        video_features = video_features[:, :length]
        audio_features = audio_features[:, :length]
        fused = self.fusion(torch.cat([video_features, audio_features], dim=-1))
        return fused

    @torch.inference_mode()
    def encode_chunk(self, audio_chunk, video_chunk, context_frames: int = 0):
        """Encode one streaming chunk, dropping the leading context frames.

        The chunk should arrive with context_frames of past frames prepended
        (frontend receptive field); the corresponding fused frames are
        trimmed so Emformer.infer sees exactly the new segment (+ any right
        context included in the chunk).
        """
        fused = self.encode_av(audio_chunk, video_chunk)
        if context_frames > 0:
            fused = fused[:, context_frames:]
        return fused

    def _step(self, batch, step_type):
        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1

        fused = self.encode_av(batch.audios, batch.videos)
        feature_lengths = torch.minimum(batch.audio_lengths, batch.video_lengths).to(device=self.device, dtype=torch.int32)
        output, src_lengths, _, _ = self.model(
            self._pad_right_context(fused),
            feature_lengths,
            prepended_targets,
            prepended_target_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=(step_type == "val"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def _pad_right_context(self, fused):
        """Emformer's non-streaming forward expects utterances right-padded
        with right_context_length extra frames and emits T outputs for a
        T+rc input; zero-pad so every fused frame is supervised/decoded."""
        if self.right_context_length > 0:
            fused = torch.nn.functional.pad(fused, (0, 0, 0, self.right_context_length))
        return fused

    def forward(self, batch, beam_width=20):
        fused = self.encode_av(batch.audios.to(self.device), batch.videos.to(self.device))
        lengths = torch.minimum(batch.audio_lengths, batch.video_lengths).to(self.device)
        hypotheses = self.decoder(self._pad_right_context(fused), lengths, beam_width=beam_width)
        return post_process_hypotheses(hypotheses, self.sp_model)[0][0]

    def stream_step(self, audio_chunk, video_chunk, state=None, hypothesis=None, beam_width=20, context_frames=0):
        fused = self.encode_chunk(
            audio_chunk.to(self.device), video_chunk.to(self.device), context_frames=context_frames
        )
        length = torch.tensor(fused.size(1), device=self.device, dtype=torch.int32)
        hypotheses, state = self.decoder.infer(
            fused.squeeze(0),
            length,
            beam_width=beam_width,
            state=state,
            hypothesis=hypothesis,
        )
        transcript = post_process_hypotheses(hypotheses, self.sp_model)[0][0]
        return transcript, hypotheses, state
