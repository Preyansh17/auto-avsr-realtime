import random

import torch
import torchvision


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, x):
        return self.functional(x)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


def _resize(size):
    return FunctionalModule(
        lambda x: x
        if x.shape[-2:] == (size, size)
        else torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    )


class VideoTransform:
    """subset: "train" (augment), "test"/"roi" (deterministic).

    frame_size selects the spatial geometry, which must be CONSISTENT across
    train, eval, and the pretrained init:
      - 88 (recipe / Conv3D frontend): crop-based geometry on 88x88 mouth ROIs
        (RandomCrop train, CenterCrop test) -- the conv frontend tolerates the
        translation jitter.
      - 44 (device / Linear frontend): the frontend is a flattened Linear and
        is position-sensitive, so every path RESIZES THE WHOLE FRAME DIRECTLY
        to 44x44 (no 88 crop), matching the pretrained device_avsr face
        pipeline and keeping train == eval. Train augmentation stays temporal
        (hflip + time mask), not spatial.
    """

    def __init__(self, subset, frame_size=88):
        div = FunctionalModule(lambda x: x / 255.0)
        gray = torchvision.transforms.Grayscale()
        norm = torchvision.transforms.Normalize(0.421, 0.165)

        if frame_size == 88:
            if subset == "train":
                self.pipeline = torch.nn.Sequential(
                    div,
                    torchvision.transforms.RandomCrop(88),
                    torchvision.transforms.RandomHorizontalFlip(0.5),
                    gray,
                    AdaptiveTimeMask(10, 25),
                    norm,
                )
            elif subset == "roi":
                self.pipeline = torch.nn.Sequential(div, _resize(88), gray, norm)
            else:
                self.pipeline = torch.nn.Sequential(div, torchvision.transforms.CenterCrop(88), gray, norm)
        else:
            # device: direct resize to frame_size for every subset
            if subset == "train":
                self.pipeline = torch.nn.Sequential(
                    div,
                    _resize(frame_size),
                    torchvision.transforms.RandomHorizontalFlip(0.5),
                    gray,
                    AdaptiveTimeMask(10, 25),
                    norm,
                )
            else:
                self.pipeline = torch.nn.Sequential(div, _resize(frame_size), gray, norm)

    def __call__(self, video):
        return self.pipeline(video)


class AudioTransform:
    def __init__(self, subset):
        self.subset = subset

    def __call__(self, audio):
        if self.subset == "train":
            return AdaptiveTimeMask(6400, 16000)(audio)
        return audio
