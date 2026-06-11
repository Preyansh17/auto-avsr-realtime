"""Streaming inference building blocks for file-based real-time AVSR.

Follows the torchaudio device_avsr tutorial (Emformer RNN-T,
RNNTBeamSearch.infer with carried state) but consumes video files
chunk-by-chunk instead of a microphone/camera StreamReader.
"""

import warnings
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torchvision

from .transforms import VideoTransform


RATE_RATIO = 640  # 16 kHz audio samples per 25 fps video frame


class SentencePieceTokenProcessor:
    """Token-id list -> text, keeping the leading word boundary (tutorial-style)."""

    def __init__(self, sp_model):
        self.sp_model = sp_model
        self.post_process_remove_list = {
            self.sp_model.unk_id(),
            self.sp_model.eos_id(),
            self.sp_model.pad_id(),
        }

    def __call__(self, tokens, lstrip: bool = True) -> str:
        filtered_hypo_tokens = [
            token_index for token_index in tokens[1:] if token_index not in self.post_process_remove_list
        ]
        output_string = "".join(self.sp_model.id_to_piece(filtered_hypo_tokens)).replace("▁", " ")
        return output_string.lstrip() if lstrip else output_string


def iter_av_windows(
    video: np.ndarray,
    audio: torch.Tensor,
    step_frames: int,
    lookback_frames: int = 0,
    lookahead_frames: int = 0,
) -> Iterator[Tuple[int, int, np.ndarray, torch.Tensor]]:
    """Yield (start, end, video_window, audio_window) over a whole recording.

    Each window covers [start - lookback, end + lookahead) frames,
    zero-padded at both edges so every window has exactly
    lookback + step + lookahead frames (audio scaled by RATE_RATIO).
    """
    total = min(video.shape[0], audio.size(0) // RATE_RATIO)
    window_frames = lookback_frames + step_frames + lookahead_frames
    for start in range(0, total, step_frames):
        end = min(start + step_frames, total)
        lo = start - lookback_frames
        hi = start + step_frames + lookahead_frames
        pad_before = max(0, -lo)
        pad_after = max(0, hi - total)
        v = video[max(lo, 0) : min(hi, total)]
        a = audio[max(lo, 0) * RATE_RATIO : min(hi, total) * RATE_RATIO]
        if pad_before:
            v = np.concatenate([np.zeros((pad_before, *v.shape[1:]), dtype=v.dtype), v])
            a = torch.cat([torch.zeros(pad_before * RATE_RATIO, a.size(1), dtype=a.dtype), a])
        if pad_after:
            v = np.concatenate([v, np.zeros((pad_after, *v.shape[1:]), dtype=v.dtype)])
            a = torch.cat([a, torch.zeros(pad_after * RATE_RATIO, a.size(1), dtype=a.dtype)])
        assert v.shape[0] == window_frames
        yield start, end, v, a


def _raw_resize_transform(size: int):
    """No-detection pipeline: /255, resize to size x size, grayscale, normalize."""
    import torch.nn as nn

    class _Resize(nn.Module):
        def forward(self, x):
            if x.shape[-2:] != (size, size):
                x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            return x

    return torch.nn.Sequential(
        torchvision.transforms.Normalize(0.0, 255.0),
        _Resize(),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Normalize(0.421, 0.165),
    )


class FacePreprocessor:
    """Tutorial preprocessing: mediapipe face detection -> alignment -> face
    crop -> resize. resize_to=44 matches the published device_avsr JIT model;
    use 88 for models trained with the recipe transforms."""

    def __init__(self, resize_to: int = 44):
        from preparation.detectors.mediapipe_face.detector import LandmarksDetector
        from preparation.detectors.mediapipe_face.video_process import VideoProcess

        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess()
        self.resize_to = resize_to
        self._fallback = _raw_resize_transform(resize_to)
        self.pipeline = torch.nn.Sequential(
            torchvision.transforms.Normalize(0.0, 255.0),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def __call__(self, video_thwc: np.ndarray) -> torch.Tensor:
        video_thwc = np.ascontiguousarray(video_thwc).astype(np.uint8)
        try:
            landmarks = self.landmarks_detector(video_thwc)
            cropped = self.video_process(video_thwc, landmarks)
        except Exception:
            cropped = None
        if cropped is None:
            warnings.warn("No face detected in chunk; falling back to raw-frame resize")
            return self._fallback(torch.from_numpy(video_thwc).permute(0, 3, 1, 2).float())
        video = torch.from_numpy(np.ascontiguousarray(cropped)).permute(0, 3, 1, 2).float()
        video = torch.stack(
            [torchvision.transforms.functional.resize(f, self.resize_to, antialias=True) for f in video]
        )
        return self.pipeline(video)


class MouthCropPreprocessor:
    """auto-avsr preprocessing: landmark detection -> mouth-ROI crop (96x96)
    -> center crop 88. Matches models fine-tuned on patient mouth-ROI data."""

    def __init__(self, detector: str = "mediapipe", device: str = "cpu"):
        if detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        elif detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector(device=device)
            self.video_process = VideoProcess(convert_gray=False)
        else:
            raise ValueError(f"Unknown detector: {detector}")
        self.transform = VideoTransform("test")
        self._fallback = VideoTransform("roi")

    def __call__(self, video_thwc: np.ndarray) -> torch.Tensor:
        video_thwc = np.ascontiguousarray(video_thwc).astype(np.uint8)
        try:
            landmarks = self.landmarks_detector(video_thwc)
            cropped = self.video_process(video_thwc, landmarks)
        except Exception:
            cropped = None
        if cropped is None:
            warnings.warn("No landmarks detected in chunk; falling back to raw-frame resize")
            return self._fallback(torch.from_numpy(video_thwc).permute(0, 3, 1, 2).float())
        video = torch.from_numpy(np.ascontiguousarray(cropped)).permute(0, 3, 1, 2).float()
        return self.transform(video)


class RoiPreprocessor:
    """For inputs that are already mouth-ROI crops (patient mp4s) or for raw
    smoke tests: no detection, resize to the model's frame size, grayscale,
    normalize. size: 88 for recipe models, 44 for the device model."""

    def __init__(self, size: int = 88):
        self.transform = _raw_resize_transform(size)

    def __call__(self, video_thwc: np.ndarray) -> torch.Tensor:
        video = torch.from_numpy(np.ascontiguousarray(video_thwc)).permute(0, 3, 1, 2).float()
        return self.transform(video)


def make_preprocessor(mode: str, detector: str = "mediapipe", face_resize: int = 44, device: str = "cpu"):
    if mode == "face":
        return FacePreprocessor(resize_to=face_resize)
    if mode == "mouth":
        return MouthCropPreprocessor(detector=detector, device=device)
    if mode in ("roi", "none"):
        # face_resize doubles as the model frame size for detection-free modes
        return RoiPreprocessor(size=face_resize)
    raise ValueError(f"Unknown preprocess mode: {mode}")


class EagerBackend:
    """Feature extraction + RNN-T from an OnlineAVSRModule checkpoint."""

    def __init__(self, module):
        self.module = module
        self.segment_length = module.segment_length
        self.right_context_length = module.right_context_length

    @property
    def rnnt(self):
        return self.module.model

    @property
    def blank_idx(self):
        return self.module.blank_idx

    def features(self, audio: torch.Tensor, video: torch.Tensor, trim_frames: int) -> torch.Tensor:
        return self.module.encode_chunk(
            audio.unsqueeze(0), video.unsqueeze(0), context_frames=trim_frames
        )


class JitBackend:
    """Feature extraction + RNN-T from the device_avsr tutorial JIT archive.

    The archive's forward(audio, video) runs frontends + fusion; its .model
    attribute is the RNN-T used by RNNTBeamSearch. The tutorial feeds
    context + buffer frames without trimming (the prepended context lands
    inside the Emformer segment and the window tail acts as right context),
    so trim_frames is ignored here.
    """

    def __init__(self, jit_module, blank_idx: int = 1023):
        self.jit = jit_module
        self.blank_idx = blank_idx
        self.segment_length = None  # opaque inside TorchScript
        self.right_context_length = None

    @property
    def rnnt(self):
        return self.jit.model

    @torch.inference_mode()
    def features(self, audio: torch.Tensor, video: torch.Tensor, trim_frames: int) -> torch.Tensor:
        return self.jit(audio.unsqueeze(0), video.unsqueeze(0))


class StreamingInferencePipeline:
    """Chunk-by-chunk AVSR with persistent RNN-T decoder state.

    carry_state=True keeps RNNTBeamSearch state and hypothesis across chunks
    (canonical streaming). carry_state=False resets per chunk and
    concatenates chunk transcripts (device_avsr tutorial behavior).
    """

    def __init__(
        self,
        backend,
        preprocessor,
        sp_model,
        beam_width: int = 10,
        carry_state: bool = True,
        device: str = "cpu",
    ):
        from torchaudio.models import RNNTBeamSearch

        self.backend = backend
        self.preprocessor = preprocessor
        self.token_processor = SentencePieceTokenProcessor(sp_model)
        self.decoder = RNNTBeamSearch(backend.rnnt, backend.blank_idx)
        self.beam_width = beam_width
        self.carry_state = carry_state
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.state = None
        self.hypotheses = None
        self.transcript = ""

    @torch.inference_mode()
    def infer_chunk(
        self,
        video_window: np.ndarray,
        audio_window: torch.Tensor,
        trim_frames: int = 0,
    ) -> Tuple[str, str, int]:
        """Run one streaming step. Returns (full transcript, new text, n_feats)."""
        video = self.preprocessor(video_window).to(self.device)
        audio = audio_window.float().to(self.device)
        feats = self.backend.features(audio, video, trim_frames)
        length = torch.tensor([feats.size(1)], device=self.device)

        if not self.carry_state:
            self.state, self.hypotheses = None, None
        self.hypotheses, self.state = self.decoder.infer(
            feats, length, self.beam_width, state=self.state, hypothesis=self.hypotheses
        )

        if self.carry_state:
            full = self.token_processor(self.hypotheses[0][0], lstrip=True)
            new = full[len(self.transcript) :] if full.startswith(self.transcript) else full
            self.transcript = full
        else:
            new = self.token_processor(self.hypotheses[0][0], lstrip=False)
            self.transcript = self.transcript + new
        return self.transcript, new, feats.size(1)


def load_eager_pipeline(
    checkpoint_path: str,
    sp_model_path: str,
    device: str = "cpu",
    preprocess: Optional[str] = None,
    detector: str = "mediapipe",
    beam_width: int = 10,
    carry_state: bool = True,
    segment_length: Optional[int] = None,
    right_context_length: Optional[int] = None,
    architecture: Optional[str] = None,
    face_resize: Optional[int] = None,
):
    """Build a streaming pipeline from an OnlineAVSRModule checkpoint.

    Architecture and segment/right-context lengths come from the checkpoint
    hparams when present; CLI overrides win.
    """
    from types import SimpleNamespace

    from .checkpoint import extract_state_dict, validate_online_state_dict
    from .module import OnlineAVSRModule
    from .text import load_sentencepiece_model

    sp_model = load_sentencepiece_model(sp_model_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {}).get("args") if isinstance(ckpt, dict) else None
    arch = architecture or getattr(hparams, "architecture", None)
    if arch is None:
        # device-variant checkpoints are recognizable by the linear video frontend
        sd_probe, _ = extract_state_dict(ckpt)
        arch = "device" if any(k.startswith("video_frontend.linear.") for k in sd_probe) else "recipe"
    default_seg, default_rc = (32, 4) if arch == "device" else (64, 0)
    seg = segment_length or getattr(hparams, "segment_length", None) or default_seg
    rc = right_context_length
    if rc is None:
        rc = getattr(hparams, "right_context_length", None)
    if rc is None:
        rc = default_rc
    module = OnlineAVSRModule(
        args=SimpleNamespace(architecture=arch, segment_length=seg, right_context_length=rc),
        sp_model=sp_model,
    )
    state_dict, _ = extract_state_dict(ckpt)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss.")}
    validate_online_state_dict(state_dict)
    module.load_state_dict(state_dict, strict=False)
    module.to(device).eval()
    if preprocess is None:
        # device model was trained on face crops; recipe models on mouth ROIs
        preprocess = "face" if arch == "device" else "roi"
    if face_resize is None:
        face_resize = 44 if arch == "device" else 88
    preprocessor = make_preprocessor(preprocess, detector=detector, face_resize=face_resize, device=device)
    pipeline = StreamingInferencePipeline(
        EagerBackend(module), preprocessor, sp_model, beam_width=beam_width,
        carry_state=carry_state, device=device,
    )
    pipeline.preprocess_mode = preprocess
    return pipeline


def load_jit_pipeline(
    jit_path: str,
    sp_model_path: str,
    device: str = "cpu",
    preprocess: str = "face",
    detector: str = "mediapipe",
    beam_width: int = 10,
    carry_state: bool = False,
    face_resize: int = 44,
):
    """Build a streaming pipeline from the device_avsr tutorial JIT model."""
    import sentencepiece as spm

    sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
    jit_module = torch.jit.load(jit_path, map_location=device)
    jit_module.eval()
    preprocessor = make_preprocessor(preprocess, detector=detector, face_resize=face_resize, device=device)
    return StreamingInferencePipeline(
        JitBackend(jit_module, blank_idx=sp_model.get_piece_size()),
        preprocessor, sp_model, beam_width=beam_width, carry_state=carry_state, device=device,
    )
