import itertools
import math
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
from .specaug import FeatureSpecAugment
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
        # "audiovisual" (default) | "audio" | "video". Unimodal builds only the
        # present frontend but KEEPS the pretrained fusion module, zero-filling
        # the absent stream so the Emformer always sees fusion-space features
        # (the modality-dropout setup the published AV model was trained with).
        # Reuses the pretrained frontend(s) + fusion + RNN-T via strict=False.
        self.modality = getattr(args, "modality", "audiovisual") or "audiovisual"
        need_audio = self.modality in ("audio", "audiovisual")
        need_video = self.modality in ("video", "audiovisual")
        if self.architecture == "device":
            self.segment_length = int(getattr(args, "segment_length", 32) or 32)
            rc = getattr(args, "right_context_length", None)
            self.right_context_length = 4 if rc is None else int(rc)
            self.audio_frontend = audio_resnet() if need_audio else None
            self.video_frontend = video_linear() if need_video else None
            self.fusion = fusion_module(hidden_dim=1024)
            self.model = emformer_rnnt_device(
                segment_length=self.segment_length,
                right_context_length=self.right_context_length,
            )
        else:
            self.segment_length = int(getattr(args, "segment_length", 64) or 64)
            self.right_context_length = int(getattr(args, "right_context_length", 0) or 0)
            self.audio_frontend = audio_resnet() if need_audio else None
            self.video_frontend = video_resnet() if need_video else None
            self.fusion = fusion_module()
            self.model = emformer_rnnt(
                segment_length=self.segment_length,
                right_context_length=self.right_context_length,
            )
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum")
        # val_loss anti-correlates with WER on these sets (low-loss epochs
        # over-emit blanks), so we also decode the val set each epoch and select
        # checkpoints by val_wer instead. Accumulators reset each val epoch.
        self.compute_val_wer = bool(getattr(args, "val_wer", True)) if args is not None else True
        self._val_wer_sum = 0.0
        self._val_wer_count = 0
        # SpecAugment in fusion-feature space (training only). Off by default;
        # --specaug turns it on. Green et al. (Interspeech 2021) dysarthric
        # recipe: heavy TIME masking on the frame axis, light CHANNEL masking on
        # the feature axis (the no-mel analog of frequency masking). Pairs with
        # the heavier waveform/video time-masking in transforms.py.
        self.specaug = FeatureSpecAugment() if getattr(args, "specaug", False) else None

    def configure_optimizers(self):
        args = self.args
        lr = float(getattr(args, "learning_rate", 8e-4) or 8e-4)
        mods = [m for m in (self.model, self.audio_frontend, self.video_frontend, self.fusion) if m is not None]
        trainable = [
            p
            for m in mods
            for p in m.parameters()
            if p.requires_grad  # LoRA fine-tunes freeze everything but the adapters
        ]
        optimizer = torch.optim.AdamW(
            trainable,
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
        # OPT-IN (--anneal-lr, default OFF). When enabled, anneal the cosine over
        # --max-steps instead of --epochs(=10000). This makes val_loss/val_wer
        # converge cleanly BUT empirically REGRESSES streaming WER (41.7% -> ~60%
        # on ROI legal): the constant-high-LR "divergent" schedule trains the
        # model hard enough to decode well in streaming mode (less future context
        # than utterance), which the low-LR tail of an annealed run never reaches.
        # So the default keeps the constant-high-LR behavior that actually wins.
        max_steps = int(getattr(args, "max_steps", 0) or 0)
        if getattr(args, "anneal_lr", False) and max_steps > 0 and steps_per_epoch > 0:
            total_epochs = max(warmup_epochs + 1, math.ceil(max_steps / steps_per_epoch))
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @property
    def decoder(self) -> RNNTBeamSearch:
        # Stored outside nn.Module attribute handling: RNNTBeamSearch wraps
        # self.model, and registering it as a submodule would duplicate every
        # RNN-T weight in state_dict (as _decoder.model.*).
        decoder = self.__dict__.get("_decoder_obj")
        if decoder is None:
            decoder = RNNTBeamSearch(self.model, self.blank_idx)
            object.__setattr__(self, "_decoder_obj", decoder)
        return decoder

    def encode_av(self, audios, videos):
        # Unimodal modes zero-fill the absent stream and run the result through
        # the pretrained fusion module, so the Emformer always sees in-distribution
        # fusion-space features. Feeding a raw 512-d frontend output straight to the
        # Emformer (the old unimodal path) is out-of-distribution and unrecoverable
        # under frozen-base LoRA, which is why audio-only never trained (val_loss
        # stuck ~38, transcripts decoupled from input). Cat order is [video, audio].
        #
        # NB: the pretrained device_avsr model was trained AUDIO-VISUAL with no
        # modality dropout (separate ASR/VSR/AV-ASR checkpoints exist upstream;
        # we bootstrap the AV one), so a zeroed stream is still off-distribution
        # for it -- zero-fill is the least-bad approximation, not a trained mode,
        # which is why audio-only (~45% WER) trails true AV (~42%).
        if self.modality == "audio":
            audio_features = self.audio_frontend(audios)
            video_features = torch.zeros_like(audio_features)
        elif self.modality == "video":
            video_features = self.video_frontend(videos)
            audio_features = torch.zeros_like(video_features)
        else:
            video_features = self.video_frontend(videos)
            audio_features = self.audio_frontend(audios)
        length = min(video_features.size(1), audio_features.size(1))
        feats = self.fusion(torch.cat([video_features[:, :length], audio_features[:, :length]], dim=-1))
        if feats.size(1) <= 0:
            raise ValueError("Frontend produced no frames")
        return feats

    def _feature_lengths(self, batch):
        if self.modality == "audio":
            return batch.audio_lengths
        if self.modality == "video":
            return batch.video_lengths
        return torch.minimum(batch.audio_lengths, batch.video_lengths)

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
        if step_type == "train" and self.specaug is not None:
            fused = self.specaug(fused)
        feature_lengths = self._feature_lengths(batch).to(device=self.device, dtype=torch.int32)
        output, src_lengths, _, _ = self.model(
            self._pad_right_context(fused),
            feature_lengths,
            prepended_targets,
            prepended_target_lengths,
        )
        # torchaudio's rnnt_loss kernel only accepts fp32/fp16 logits and
        # rejects bfloat16 ("logits must be float32 or float16"), so bf16-mixed
        # AMP crashes here. Cast to fp32 for the loss: required for bf16-mixed,
        # a no-op for fp32, and strictly safer for fp16 (loss computed in fp32).
        loss = self.loss(output.float(), batch.targets, src_lengths, batch.target_lengths)
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=(step_type == "val"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def on_validation_epoch_start(self):
        self._val_wer_sum = 0.0
        self._val_wer_count = 0

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if self.compute_val_wer:
            self._accumulate_val_wer(batch)
        return loss

    @torch.no_grad()
    def _accumulate_val_wer(self, batch):
        """Greedy-decode each val clip and accumulate WER vs the reference.

        WER (not val_loss) is the checkpoint-selection signal: val_loss bottoms
        out while the model still emits blanks, so loss-best != WER-best. Greedy
        (beam_width=1) keeps this cheap; relative ranking tracks beam decoding.
        """
        from .text import compute_wer

        for i in range(batch.targets.size(0)):
            try:
                single = AVBatch(
                    audios=batch.audios[i : i + 1],
                    videos=batch.videos[i : i + 1],
                    audio_lengths=batch.audio_lengths[i : i + 1],
                    video_lengths=batch.video_lengths[i : i + 1],
                    targets=batch.targets[i : i + 1],
                    target_lengths=batch.target_lengths[i : i + 1],
                )
                hypothesis = self.forward(single, beam_width=1)
                ref_ids = batch.targets[i, : batch.target_lengths[i]].tolist()
                reference = self.sp_model.decode([int(t) for t in ref_ids])
                self._val_wer_sum += compute_wer(reference, hypothesis)
                self._val_wer_count += 1
            except Exception:
                continue

    def on_validation_epoch_end(self):
        # Monotonic epoch counter so a ModelCheckpoint can keep the last-N
        # CONSECUTIVE epochs (for --avg-last-n endpoint averaging) rather than
        # the top-k by val_loss.
        self.log("epoch_idx", float(self.current_epoch))
        if self.compute_val_wer and self._val_wer_count > 0:
            # Single-GPU runs; manual mean is fine (no cross-rank reduction).
            self.log("val_wer", self._val_wer_sum / self._val_wer_count, prog_bar=True)

    def _pad_right_context(self, fused):
        """Emformer's non-streaming forward expects utterances right-padded
        with right_context_length extra frames and emits T outputs for a
        T+rc input; zero-pad so every fused frame is supervised/decoded."""
        if self.right_context_length > 0:
            fused = torch.nn.functional.pad(fused, (0, 0, 0, self.right_context_length))
        return fused

    def forward(self, batch, beam_width=20):
        fused = self.encode_av(batch.audios.to(self.device), batch.videos.to(self.device))
        lengths = self._feature_lengths(batch).to(self.device)
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
