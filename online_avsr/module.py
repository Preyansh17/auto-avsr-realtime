import itertools
from collections import namedtuple

import torch
import torchaudio
from pytorch_lightning import LightningModule
from torchaudio.models import RNNTBeamSearch

from . import EXPECTED_SPM_VOCAB_SIZE
from .models import audio_resnet, emformer_rnnt, fusion_module, video_resnet
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
        self.audio_frontend = audio_resnet()
        self.video_frontend = video_resnet()
        self.fusion = fusion_module()
        self.model = emformer_rnnt()
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum")
        lr = float(getattr(args, "learning_rate", 8e-4))
        self.optimizer = torch.optim.AdamW(
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

    def configure_optimizers(self):
        return self.optimizer

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

    def _step(self, batch, step_type):
        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1

        fused = self.encode_av(batch.audios, batch.videos)
        feature_lengths = torch.minimum(batch.audio_lengths, batch.video_lengths).to(device=self.device, dtype=torch.int32)
        output, src_lengths, _, _ = self.model(
            fused,
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

    def forward(self, batch, beam_width=20):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        fused = self.encode_av(batch.audios.to(self.device), batch.videos.to(self.device))
        lengths = torch.minimum(batch.audio_lengths, batch.video_lengths).to(self.device)
        hypotheses = decoder(fused, lengths, beam_width=beam_width)
        return post_process_hypotheses(hypotheses, self.sp_model)[0][0]

    def stream_step(self, audio_chunk, video_chunk, state=None, hypothesis=None, beam_width=20):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        fused = self.encode_av(audio_chunk.to(self.device), video_chunk.to(self.device))
        length = torch.tensor(fused.size(1), device=self.device, dtype=torch.int32)
        hypotheses, state = decoder.infer(
            fused.squeeze(0),
            length,
            beam_width=beam_width,
            state=state,
            hypothesis=hypothesis,
        )
        transcript = post_process_hypotheses(hypotheses, self.sp_model)[0][0]
        return transcript, hypotheses, state
