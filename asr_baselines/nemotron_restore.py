#!/usr/bin/env python3
"""Load a Nemotron .nemo checkpoint, including adapter/LoRA checkpoints.

Plain full-finetune checkpoints restore fine through NeMo's normal
`ASRModel.restore_from()`. Adapter checkpoints (produced by
`nemotron_finetune.py --lora`) do not, and the failure is doubly confusing:

  * via the abstract base -- `ASRModel.restore_from(path)` -- you get
    `TypeError: Can't instantiate abstract class ASRModel with abstract
    methods setup_training_data, setup_validation_data`, which looks like a
    bad checkpoint or a bad call but is really NeMo failing to dispatch to
    the concrete subclass and falling back to the base;
  * via the concrete class -- `EncDecRNNTBPEModel.restore_from(path)` -- the
    real cause surfaces: `AttributeError: 'ConformerEncoder' object has no
    attribute 'add_adapter'`.

Root cause (confirmed on nemo_toolkit 2.7.3, 2026-07-25): a saved adapter
checkpoint carries an `adapters` section in its config, so
`EncDecRNNTBPEModel.__init__` calls `setup_adapters()`, which tries to attach
adapters to `self.encoder` -- but at that point the encoder is still a plain
`ConformerEncoder`. Nothing has yet swapped it for the adapter-capable
`ConformerEncoderAdapter` subclass, because the call that does that
(`replace_adapter_compatible_modules()`) is something the *caller* is expected
to make, and it cannot run before `__init__` has finished.

Workaround here: restore with the `adapters` section stripped from the config
so `__init__` never attempts the attach, then make the encoder
adapter-compatible, re-declare the adapters from the saved config, and load
the adapter weights back out of the checkpoint's own state dict.

Verified on a LoRA smoketest checkpoint: 96 adapter tensors present in the
file, restored with `missing=0, unexpected=0` (i.e. nothing silently dropped)
and `get_enabled_adapters() == ['lora']`. A restore that merely *constructs*
without error is not a pass -- it has to bring the adapter weights back too.
"""

import dataclasses
import os
import tarfile
import tempfile


def _has_adapters(nemo_path):
    """True if the .nemo's config declares an `adapters` section."""
    try:
        with tarfile.open(nemo_path, "r:*") as tar:
            names = [m for m in tar.getnames() if m.endswith("model_config.yaml")]
            if not names:
                return False
            text = tar.extractfile(names[0]).read().decode("utf-8", "replace")
        return any(line.rstrip().startswith("adapters:") for line in text.splitlines())
    except Exception:
        # Not readable as a tar, or no config -- let the normal restore path
        # produce the real error rather than masking it here.
        return False


def _restore_adapter_checkpoint(nemo_path, tmp_dir=None):
    import torch
    from omegaconf import OmegaConf, open_dict
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig

    with tempfile.TemporaryDirectory(dir=tmp_dir, ignore_cleanup_errors=True) as td:
        with tarfile.open(nemo_path, "r:*") as tar:
            tar.extractall(td)

        cfg_path = os.path.join(td, "model_config.yaml")
        cfg = OmegaConf.load(cfg_path)
        adapters_cfg = OmegaConf.to_container(cfg.adapters, resolve=True)
        with open_dict(cfg):
            del cfg["adapters"]
        OmegaConf.save(cfg, cfg_path)

        # strict=False: the checkpoint's adapter tensors have nowhere to go yet.
        model = EncDecRNNTBPEModel.restore_from(
            nemo_path, override_config_path=cfg_path, strict=False
        )
        # Remember where NeMo put the model. add_adapter() builds fresh modules
        # and the adapter weights below are read with map_location="cpu", so
        # without re-homing at the end the model ends up split across devices
        # and inference dies with "Expected all tensors to be on the same
        # device". Captured here rather than assumed, so a CPU-only host still
        # works.
        device = next(model.parameters()).device

        # Swap ConformerEncoder -> ConformerEncoderAdapter (same weights).
        model.replace_adapter_compatible_modules()

        # `global_cfg` is adapter-system metadata, not an adapter. Saved adapter
        # entries also carry bookkeeping keys (e.g. adapter_meta_cfg) that
        # LinearAdapterConfig's constructor rejects, so filter to real fields.
        fields = {f.name for f in dataclasses.fields(LinearAdapterConfig)}
        names = [k for k in adapters_cfg if not k.startswith("__") and k != "global_cfg"]
        for name in names:
            entry = {k: v for k, v in adapters_cfg[name].items() if k in fields}
            model.add_adapter(name=name, cfg=LinearAdapterConfig(**entry))

        weights = [f for f in os.listdir(td) if f.endswith(".ckpt")]
        if not weights:
            raise RuntimeError(f"no weights file inside {nemo_path}")
        state = torch.load(os.path.join(td, weights[0]), map_location="cpu",
                           weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        n_adapter = sum(1 for k in model.state_dict() if "adapter" in k)
        if n_adapter == 0:
            raise RuntimeError(
                f"{nemo_path} declared adapters but none survived the restore "
                f"-- refusing to return a silently-adapterless model"
            )
        model = model.to(device)
        print(f"restored adapter checkpoint: {n_adapter} adapter tensors, "
              f"enabled={model.get_enabled_adapters()}, "
              f"missing={len(missing)} unexpected={len(unexpected)}, "
              f"device={device}")
        return model


def restore_asr_model(model_arg, tmp_dir=None):
    """Load `model_arg` as a local .nemo path or a pretrained hub id.

    Routes adapter/LoRA checkpoints through the workaround above and
    everything else through NeMo's normal path.
    """
    import nemo.collections.asr as nemo_asr

    if model_arg.endswith(".nemo") and os.path.isfile(model_arg):
        if _has_adapters(model_arg):
            return _restore_adapter_checkpoint(model_arg, tmp_dir=tmp_dir)
        return nemo_asr.models.ASRModel.restore_from(model_arg)
    return nemo_asr.models.ASRModel.from_pretrained(model_arg)
