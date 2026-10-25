#!/usr/bin/env python3
"""Finetune NVIDIA Nemotron streaming ASR on patient dysarthric speech.

Model: nvidia/nemotron-speech-streaming-en-0.6b -- a cache-aware
FastConformer-RNNT (24 encoder layers), the streaming RNN-T analog to Green et
al.'s finetuned RNN-T. This is the streaming audio baseline the AV Emformer
must beat.

Green recipe applied here:
  * train only the first N FastConformer encoder layers (default 5), freeze the
    rest of the encoder + the decoder/joint;
  * SpecAugment with cut frequency masking, blown-up time masking (NeMo's native
    spec_augment block, set from CLI -> mirrors specaug_green.yaml).

Also supports --lora: freezes the whole model and trains only a small
bottleneck adapter inserted into each encoder layer (NeMo's built-in
LinearAdapterConfig -- a Houlsby-style residual adapter, the closest
equivalent NeMo ships for RNNT-BPE conformer encoders; not the same
mechanism as the AV Emformer's true low-rank LoRA in online_avsr/lora.py,
but same intent: frozen backbone, small trainable add-on).

NeMo's API shifts between releases; the version-sensitive spots are marked. Pin
via requirements-nemotron.txt and run in its OWN env (NeMo pins a torch that
clashes with the AV torch-2.6 env).

  python -m asr_baselines.nemotron_finetune \
    --train-manifest train.jsonl --val-manifest val.jsonl \
    --output-dir $EXP/nemotron_green --epochs 30 --unfreeze-encoder-layers 5
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr-schedule", choices=["warmup_flat", "warmup_cosine", "noam"], default="warmup_flat",
                   help="warmup_flat: linear warmup over --warmup-steps optimizer steps, then constant "
                        "--learning-rate (NeMo WarmupPolicy). warmup_cosine: same warmup, cosine decay to "
                        "--min-lr. noam: the pretrained config's NoamAnnealing (d_model 1024, min_lr 1e-6), "
                        "which scales --learning-rate by d_model^-0.5 * warmup^-0.5 (~2.2e-3 at warmup 200): "
                        "lr 1e-4 peaks at 2.2e-7 and is then clamped to min_lr, so every run before "
                        "2026-10-20 trained at an effective flat 1e-6 whatever --learning-rate said. "
                        "Use noam only to reproduce those runs.")
    p.add_argument("--min-lr", type=float, default=0.0, help="floor for warmup_cosine")
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--decoder-joint-lr-scale", type=float, default=1.0,
                   help="scale factor applied to --learning-rate for the "
                        "decoder+joint param group only (encoder and "
                        "everything else keeps the base LR); 1.0 = disabled, "
                        "single flat LR for the whole model (default). Uses "
                        "NeMo's native model.cfg.optim_param_groups mechanism.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-duration", type=float, default=20.0)
    p.add_argument("--min-duration", type=float, default=0.3)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--resume-ckpt", default=None,
                   help="PTL .ckpt path to resume optimizer/scheduler/epoch state from")
    p.add_argument("--speed-perturb", action="store_true",
                   help="enable Lhotse's train_ds.perturb_speed (random speed "
                        "perturbation augmentation), off by default")
    p.add_argument("--ckpt-every-n-epochs", type=int, default=1,
                   help="write Lightning's resume checkpoint (.ckpt, ~7GB -- "
                        "3x the model because it carries Adam's two moments "
                        "per param) only every N epochs. This write is pure "
                        "resume insurance for the cluster's auto-cancellation "
                        "policy; it is NOT the best-checkpoint deliverable "
                        "(that's SaveBestWER's .nemo). At ~171MB/s on scratch "
                        "a 7GB write is ~40s -- on par with a whole epoch's "
                        "training compute, so this is the single biggest "
                        "wall-clock/GPU-utilization lever. Cost of N>1: a "
                        "cancellation loses up to N epochs of progress.")
    p.add_argument("--val-every-n-epochs", type=int, default=1,
                   help="run validation (and therefore best-checkpoint "
                        "selection + checkpoint writes) only every N epochs. "
                        "At this dataset's size the per-epoch validation pass "
                        "and its multi-GB checkpoint writes dominate wall-clock "
                        "over actual training compute, so N>1 is the main "
                        "speed lever. Costs checkpoint-selection granularity: "
                        "N=3 over 210 epochs gives 70 candidate checkpoints "
                        "instead of 210.")
    # Green freeze
    p.add_argument("--unfreeze-encoder-layers", type=int, default=5,
                   help="train encoder layers [0, N); -1 = train all")
    p.add_argument("--freeze-decoder", action="store_true", default=True)
    p.add_argument("--train-decoder", dest="freeze_decoder", action="store_false")
    # LoRA (NeMo's bottleneck adapter on the FastConformer encoder -- Houlsby-
    # style residual adapter, not a literal low-rank weight decomposition like
    # the AV Emformer's own online_avsr/lora.py; NeMo doesn't expose true LoRA
    # for RNNT-BPE conformer encoders, this is the closest built-in equivalent
    # and serves the same purpose: frozen backbone, small trainable add-on).
    # Mutually exclusive with --unfreeze-encoder-layers/--train-decoder; when
    # set, those are ignored (encoder+decoder+joint all frozen, only the
    # adapter is trainable).
    p.add_argument("--lora", action="store_true",
                   help="freeze the whole model, train only a bottleneck adapter "
                        "inserted into each FastConformer encoder layer")
    p.add_argument("--lora-dim", type=int, default=32,
                   help="adapter bottleneck dimension")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    # Plain --lora freezes the RNN-T decoder and joint along with the encoder,
    # so the entire healthy->dysarthric domain shift has to be absorbed by
    # encoder-side adapters. The dim=32 LR probe (2026-07-31) suggests that is
    # the binding constraint, not the LR: 1e-4/5e-4/1e-3 landed within 0.9pp of
    # each other (0.2679/0.2768/0.2768 val_wer) while full finetune reached
    # 0.1875 in half the epochs. This flag keeps the frozen encoder but lets the
    # decoder+joint adapt, which separates "adapter capacity is too small" from
    # "the frozen decoder is what hurts".
    p.add_argument("--lora-train-decoder", action="store_true",
                   help="with --lora: also train the RNN-T decoder and joint "
                        "(encoder still frozen apart from its adapters)")
    # Cache-aware streaming context (left,right) in frames; keep model default if unset.
    p.add_argument("--att-context-size", type=int, nargs=2, default=None,
                   help="e.g. 70 13 ; default keeps the pretrained streaming context")
    # SpecAugment (NeMo native). Green-leaning: heavy time, light freq.
    p.add_argument("--no-specaug", action="store_true")
    p.add_argument("--freq-masks", type=int, default=1)
    p.add_argument("--freq-width", type=int, default=13)
    p.add_argument("--time-masks", type=int, default=10)
    p.add_argument("--time-width", type=float, default=0.05, help="fraction of frames per time mask")
    p.add_argument("--seed", type=int, default=int(os.environ["SEED"]) if os.environ.get("SEED") else None,
                   help="Seed all RNG (lightning.pytorch.seed_everything) for reproducible runs. "
                        "Same rationale as train.py/whisper_finetune.py -- this repo's recipes have "
                        "shown large unseeded run-to-run WER variance; seed + multi-seed runs are "
                        "needed to measure anything.")
    return p.parse_args()


def make_lr_log_callback(warmup_steps):
    """Print the optimizer's actual learning rate at a few optimizer steps, so the effective schedule is in the log
    (the configured --learning-rate was silently not the effective one until 2026-10-20)."""
    import lightning.pytorch as pl

    marks = {1, warmup_steps // 2, warmup_steps, warmup_steps + 1, 2 * warmup_steps}

    class LRLog(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            s = trainer.global_step
            if (s in marks or (s > 0 and s % 5000 == 0)) and getattr(self, "_last", None) != s:
                self._last = s
                lrs = sorted({f"{g['lr']:.3g}" for g in trainer.optimizers[0].param_groups})
                print(f"[lr] optimizer step {s}: lr={','.join(lrs)}", flush=True)

    return LRLog()


def make_save_best_wer_callback(out_path):
    """Lightning callback (built lazily, after `lightning` is imported, to
    keep this script's imports fast for --help): saves a .nemo copy whenever
    val_wer improves.

    NeMo's RNNT models compute val_wer every validation epoch (rnnt_models.py
    multi_validation_epoch_end), but nothing in this script ever selected on
    it -- model.save_to() was called unconditionally after trainer.fit(),
    i.e. always the LAST epoch, no best-checkpoint tracking at all (unlike
    Whisper's load_best_model_at_end). This mirrors that HF Trainer behavior
    using NeMo's own save_to() format directly (avoids depending on Lightning
    .ckpt restore semantics for a NeMo ModelPT, a separate, less-proven code
    path in this stack). Degrades safely: if val_wer never shows up in
    trainer.callback_metrics (untested plumbing on this NeMo version),
    best_wer stays inf and the caller's fallback (last-epoch-only) kicks in
    -- printed clearly, not a silent no-op.

    best_wer is persisted via Lightning's callback state hooks. Without that
    it is a plain Python attribute that resets to inf on --resume-ckpt, so the
    first post-resume validation epoch unconditionally overwrites a good
    nemotron_best.nemo with whatever that epoch produced, even if worse. That
    bit this project repeatedly during the 2026-07-21/24 cluster-cancellation
    resumes and had to be worked around by hand-copying the .nemo before every
    resume; state_dict/load_state_dict makes the workaround unnecessary.
    """
    from lightning.pytorch import Callback

    class SaveBestWER(Callback):
        def __init__(self):
            self.out_path = out_path
            self.best_wer = float("inf")

        def state_dict(self):
            return {"best_wer": self.best_wer}

        def load_state_dict(self, state_dict):
            self.best_wer = state_dict.get("best_wer", float("inf"))
            print(f"[resume] SaveBestWER restored best_wer={self.best_wer:.4f} "
                  f"-- will only overwrite {self.out_path} on a genuine improvement")

        # on_validation_end, not on_validation_epoch_end: Lightning runs callback epoch-end hooks before the
        # module's, and NeMo logs val_wer in the module hook, so at epoch-end callback_metrics still held the
        # previous epoch's val_wer and the saved weights were one epoch past the best.
        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            wer = trainer.callback_metrics.get("val_wer")
            if wer is None:
                return
            wer = float(wer)
            if wer < self.best_wer:
                self.best_wer = wer
                pl_module.save_to(self.out_path)
                print(f"[best] epoch={trainer.current_epoch} val_wer={wer:.4f} -> saved {self.out_path}")

    return SaveBestWER()


def set_spec_augment(model, args):
    from omegaconf import open_dict

    if args.no_specaug:
        # Replace with identity by zeroing masks.
        with open_dict(model.cfg):
            model.cfg.spec_augment.freq_masks = 0
            model.cfg.spec_augment.time_masks = 0
        model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)
        print("SpecAugment: DISABLED (ablation)")
        return
    with open_dict(model.cfg):
        sa = model.cfg.spec_augment
        sa.freq_masks = args.freq_masks
        sa.freq_width = args.freq_width
        sa.time_masks = args.time_masks
        sa.time_width = args.time_width
    model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)
    print(f"SpecAugment: freq(masks={args.freq_masks}, width={args.freq_width}) "
          f"time(masks={args.time_masks}, width={args.time_width})")


def apply_freeze(model, args):
    n = args.unfreeze_encoder_layers
    if n == -1 and not args.freeze_decoder:
        print("Freeze: full finetune")
        return
    for prm in model.parameters():
        prm.requires_grad = False
    # FastConformer encoder layers live in model.encoder.layers (ConformerLayer list).
    layers = model.encoder.layers
    train_idx = range(len(layers)) if n == -1 else range(min(n, len(layers)))
    for i in train_idx:
        for prm in layers[i].parameters():
            prm.requires_grad = True
    if not args.freeze_decoder:
        for mod in (model.decoder, model.joint):
            for prm in mod.parameters():
                prm.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Freeze: training encoder layers {list(train_idx)} "
          f"(decoder frozen={args.freeze_decoder}); "
          f"trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total:.1%})")


def apply_lora(model, args):
    """Freeze everything, then attach a bottleneck adapter (NeMo's
    LinearAdapterConfig, Houlsby-style: down-proj -> swish -> up-proj,
    zero-init'd output so it starts as a no-op residual) to every FastConformer
    encoder layer. Only the adapter params end up trainable.

    The base EncDecRNNTBPEModel's encoder (plain ConformerEncoder) doesn't
    implement the adapter mixin -- replace_adapter_compatible_modules() swaps
    it in-place for ConformerEncoderAdapter (same weights, adapter-capable
    subclass) before add_adapter() can be called.

    Loading the result back is NOT symmetric. An earlier version of this
    docstring claimed NeMo's own restore_from() reconstructs the adapters from
    the saved .nemo's cfg.adapters section, so the eval scripts needed no
    special-casing. That is wrong: setup_adapters() runs inside
    EncDecRNNTBPEModel.__init__, before anything has swapped the encoder for
    the adapter-capable subclass, so it dies with "'ConformerEncoder' object
    has no attribute 'add_adapter'". Every eval path therefore goes through
    asr_baselines/nemotron_restore.py, which strips cfg.adapters, restores,
    swaps the encoder, re-declares the adapters, and reloads their weights.
    """
    from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig

    for prm in model.parameters():
        prm.requires_grad = False
    model.replace_adapter_compatible_modules()
    cfg = LinearAdapterConfig(
        in_features=model.cfg.encoder.d_model,
        dim=args.lora_dim,
        dropout=args.lora_dropout,
    )
    model.add_adapter(name="lora", cfg=cfg)
    model.unfreeze_enabled_adapters()
    # Order matters: unfreeze_enabled_adapters() is what re-enables grads after
    # the blanket freeze above, so the decoder/joint unfreeze has to come after
    # it, not before.
    if args.lora_train_decoder:
        for module in (model.decoder, model.joint):
            for prm in module.parameters():
                prm.requires_grad = True
    scope = ("decoder+joint TRAINABLE" if args.lora_train_decoder
             else "decoder+joint frozen")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: bottleneck adapter dim={args.lora_dim} on all encoder layers "
          f"({scope}); "
          f"trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total:.1%})")


def main():
    args = parse_args()
    if args.seed is not None:
        from lightning.pytorch import seed_everything

        seed_everything(args.seed, workers=True)
        print(f"seed_everything({args.seed}, workers=True)")
    import nemo.collections.asr as nemo_asr
    # NeMo's model classes inherit from lightning.pytorch.LightningModule (the
    # unified "lightning" package), NOT the standalone pytorch_lightning
    # package -- despite similar naming/history, newer versions of the two
    # define distinct, non-aliased LightningModule classes. pl.Trainer.fit()
    # from the wrong package raises "model must be a LightningModule ... got
    # EncDecRNNTBPEModel" since the isinstance check fails across packages.
    # Confirmed via ASRModel.__mro__ on the installed nemo_toolkit 2.7.3.
    import lightning.pytorch as pl
    from omegaconf import OmegaConf, open_dict

    os.makedirs(args.output_dir, exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)

    # NOTE on RNNT loss backend: NeMo's default (warprnnt_numba) originally
    # crashed with a numba CUDA-JIT "Signature mismatch" error. Root cause was
    # numba 0.66.0 (unpinned by nemo_toolkit, resolved to the bleeding-edge
    # latest -- see requirements-nemotron.txt) not being vetted against NeMo's
    # CUDA kernel code; fixed by pinning numba==0.60.0. Left as the default
    # (warprnnt_numba) here, which is what NeMo actually tests against; a
    # loss_name="pytorch" swap was tried and reverted (still numba-JIT'd under
    # a different name, not actually pure PyTorch -- didn't help).

    # After the numba pin, training crashed with a hard segfault (not a Python
    # exception) at 0 iterations/0 seconds, exactly at the eval->train mode
    # transition following sanity-check validation. Ruled out precision/AMP as
    # the cause (identical segfault under --precision 32-true). The remaining
    # suspect: CUDA graph capture/replay for the greedy RNNT decoder
    # (GreedyBatchedRNNTInfer), which the logs show toggling "enabled"/
    # "disabled" right around the crash point -- CUDA graphs are a much more
    # version/driver-sensitive feature than plain CUDA kernels. Disabled via
    # the documented use_cuda_graph_decoder config flag.
    with open_dict(model.cfg):
        if "decoding" in model.cfg and "greedy" in model.cfg.decoding:
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(model.cfg.decoding)

    # Data
    #
    # The pretrained model's saved train_ds/validation_ds config has several
    # bucketing-related keys set to null (num_buckets, bucket_buffer_size,
    # bucket_duration_bins, bucket_batch_size, batch_duration, max_tps,
    # shuffle_buffer_size). Newer NeMo's LhotseDataLoadingConfig schema types
    # these as bare (non-Optional) ints/floats, so OmegaConf.merge rejects each
    # None value one at a time at schema-validation time (confirmed by hitting
    # num_buckets first, then bucket_buffer_size next -- whack-a-mole if fixed
    # one at a time). Deleting the null keys entirely lets the dataclass
    # schema's own defaults fill in during the merge instead of an explicit
    # None override. Bucketing itself is a throughput optimization for large
    # multi-hour corpora, irrelevant for this ~268-clip patient set.
    _bucketing_keys = (
        "num_buckets", "bucket_buffer_size", "bucket_duration_bins",
        "bucket_batch_size", "batch_duration", "max_tps", "shuffle_buffer_size",
    )
    with open_dict(model.cfg):
        model.cfg.train_ds.manifest_filepath = args.train_manifest
        model.cfg.train_ds.batch_size = args.batch_size
        model.cfg.train_ds.num_workers = args.num_workers
        model.cfg.train_ds.max_duration = args.max_duration
        model.cfg.train_ds.min_duration = args.min_duration
        model.cfg.train_ds.shuffle = True
        model.cfg.train_ds.use_bucketing = False
        model.cfg.train_ds.perturb_speed = args.speed_perturb
        # The pretrained model's saved train_ds/validation_ds config default to
        # text_field="answer" (visible in the printed config at model-load
        # time) -- some fine-tuning use case's manifest convention, not the
        # standard ASR "text" field. make_nemo_manifest.py writes
        # {"audio_filepath", "duration", "text"}; without this override, every
        # training/validation example's target transcript is silently missing,
        # and both freeze and full-finetune configs (3 seeds each, 6 runs
        # total) collapsed to identical WER=102.7568% (predicting only "⁇"
        # unknown-token placeholders for every input) -- a data-wiring bug, not
        # a training-dynamics one.
        model.cfg.train_ds.text_field = "text"
        model.cfg.validation_ds.manifest_filepath = args.val_manifest
        model.cfg.validation_ds.text_field = "text"
        model.cfg.validation_ds.batch_size = args.batch_size
        model.cfg.validation_ds.num_workers = args.num_workers
        model.cfg.validation_ds.shuffle = False
        model.cfg.validation_ds.use_bucketing = False
        for key in _bucketing_keys:
            model.cfg.train_ds.pop(key, None)
            model.cfg.validation_ds.pop(key, None)
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    # Streaming context (optional)
    if args.att_context_size is not None and hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size(list(args.att_context_size))
        print(f"att_context_size set to {args.att_context_size}")

    set_spec_augment(model, args)
    if args.lora:
        apply_lora(model, args)
    else:
        apply_freeze(model, args)

    # Optimizer / schedule
    #
    # NeMo's scheduler setup auto-derives max_steps from len(train_dataloader.
    # dataset) when not given explicitly -- but Lhotse's dynamic/streaming
    # dataset (LhotseSpeechToTextBpeDataset) has no __len__ by design (lazy
    # sampling), so that auto-derivation crashes with "TypeError: object of
    # type 'LhotseSpeechToTextBpeDataset' has no len()" the moment the
    # Trainer's own configure_optimizers() hook re-runs setup_optimization()
    # internally. Compute max_steps ourselves from the manifest line count
    # (a reasonable approximation -- Lhotse's duration-based dynamic batching
    # means the true per-epoch step count varies slightly, but the scheduler
    # only needs a sane nonzero target, not an exact one) and set it
    # explicitly so NeMo never needs the dataset length.
    with open(args.train_manifest, encoding="utf-8") as f:
        num_train_examples = sum(1 for line in f if line.strip())
    steps_per_epoch = max(1, num_train_examples // args.batch_size)
    max_steps = steps_per_epoch * args.epochs
    with open_dict(model.cfg):
        model.cfg.optim.lr = args.learning_rate
        model.cfg.optim.weight_decay = args.weight_decay
        # Force the plain (non-fused, non-foreach) AdamW kernel path. With
        # foreach/fused left at PyTorch's auto-selected default (None, None in
        # the logged optimizer config), training segfaulted with a hard crash
        # (not a Python exception) inside torch/optim/adam.py's step() --
        # confirmed via a faulthandler traceback after ruling out the RNNT
        # loss backend, precision/AMP, CUDA graphs, and dataloader workers as
        # causes (identical crash under all of those). On this stack (torch
        # 2.12.1+cu130 -- a very recent build), the auto-selected fused/foreach
        # CUDA Adam kernel is the remaining suspect; the plain per-parameter
        # Python-loop implementation avoids that code path entirely.
        model.cfg.optim.foreach = False
        model.cfg.optim.fused = False
        # The scheduler steps once per OPTIMIZER step (Lightning interval "step"), i.e. per grad_accum batches.
        opt_steps = max(1, max_steps // args.grad_accum)
        if args.lr_schedule == "noam":
            if "sched" in model.cfg.optim and model.cfg.optim.sched is not None:
                model.cfg.optim.sched.warmup_steps = args.warmup_steps
                model.cfg.optim.sched.max_steps = max_steps
        elif args.lr_schedule == "warmup_flat":
            # WarmupPolicy drops to min_lr past max_steps; the step estimate above is approximate (Lhotse dynamic
            # batching), so give it 2x headroom rather than risk a tail at lr 0.
            model.cfg.optim.sched = OmegaConf.create({"name": "WarmupPolicy", "warmup_steps": args.warmup_steps,
                                                      "max_steps": 2 * opt_steps, "min_lr": 0.0})
        else:
            model.cfg.optim.sched = OmegaConf.create({"name": "CosineAnnealing", "warmup_steps": args.warmup_steps,
                                                      "max_steps": opt_steps, "min_lr": args.min_lr})
        # Differential decoder/joint LR: NeMo's ModelPT.setup_optimizer_param_groups
        # (called internally by setup_optimization) natively reads
        # cfg.optim_param_groups -- a dict keyed by top-level model attribute
        # name (must match e.g. "decoder"/"joint" exactly, matched by
        # named_parameters() prefix), each value a dict of per-group optimizer
        # kwargs (lr, weight_decay, etc). Unlisted params (encoder + anything
        # else) fall back to the top-level optim.lr set above. No monkeypatch
        # needed -- this is the documented extension point, not a private API.
        if args.decoder_joint_lr_scale != 1.0:
            decoder_joint_lr = args.learning_rate * args.decoder_joint_lr_scale
            model.cfg.optim_param_groups = {
                "decoder": {"lr": decoder_joint_lr},
                "joint": {"lr": decoder_joint_lr},
            }
    if args.decoder_joint_lr_scale != 1.0:
        print(f"Differential LR: encoder/other @ lr={args.learning_rate}, "
              f"decoder+joint @ lr={args.learning_rate * args.decoder_joint_lr_scale} "
              f"(scale={args.decoder_joint_lr_scale})")
    model.setup_optimization(model.cfg.optim)
    print(f"train examples={num_train_examples} steps_per_epoch={steps_per_epoch} max_steps={max_steps} "
          f"(optimizer steps ~{max_steps // args.grad_accum}) lr_schedule={args.lr_schedule} "
          f"sched={OmegaConf.to_container(model.cfg.optim.sched) if model.cfg.optim.get('sched') else None}")

    best_out = os.path.join(args.output_dir, "nemotron_best.nemo")
    best_wer_cb = make_save_best_wer_callback(best_out)

    # Validation frequency is the main wall-clock lever at this dataset size
    # (314 train clips): a training epoch is only ~10-15s of actual compute,
    # but every validation epoch also drags in SaveBestWER's 2.4GB .nemo
    # serialization (whenever val_wer improves -- i.e. most early epochs) and
    # Lightning's own default ModelCheckpoint .ckpt write. Those I/O stalls,
    # not batch compute, are what keep averaged GPU utilization low. Raising
    # this reduces all three costs at once. Batch size was tried first and
    # barely moved epochs/minute, which is what pointed here.
    if args.val_every_n_epochs > 1:
        print(f"Validation every {args.val_every_n_epochs} epochs "
              f"(~{args.epochs // args.val_every_n_epochs} checkpoint-selection "
              f"points over {args.epochs} epochs)")

    callbacks = [best_wer_cb, make_lr_log_callback(args.warmup_steps)]
    if args.ckpt_every_n_epochs > 1:
        # Replace Lightning's implicit every-validation-epoch ModelCheckpoint
        # with an explicit, less frequent one. Passing any ModelCheckpoint in
        # callbacks suppresses the default. save_top_k=-1 with every_n_epochs
        # would accumulate 7GB files; save_top_k=1 keeps exactly one rolling
        # resume point, which is all --resume-ckpt needs.
        from lightning.pytorch.callbacks import ModelCheckpoint

        callbacks.append(ModelCheckpoint(
            every_n_epochs=args.ckpt_every_n_epochs,
            save_top_k=1,
            monitor=None,
            save_last=False,
        ))
        print(f"Resume-checkpoint (.ckpt) written every "
              f"{args.ckpt_every_n_epochs} epochs instead of every validation "
              f"epoch (~7GB/write at ~171MB/s scratch throughput)")

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=args.grad_accum,
        check_val_every_n_epoch=args.val_every_n_epochs,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir=args.output_dir,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )
    model.set_trainer(trainer)
    trainer.fit(model, ckpt_path=args.resume_ckpt)

    out = os.path.join(args.output_dir, "nemotron_patient.nemo")
    model.save_to(out)
    print(f"Saved finetuned model (last epoch) -> {out}")
    if best_wer_cb.best_wer < float("inf"):
        print(f"Saved finetuned model (best val_wer={best_wer_cb.best_wer:.4f}) -> {best_out}")
    else:
        print("WARNING: val_wer never appeared in trainer.callback_metrics -- "
              "best-checkpoint selection did not engage, only last-epoch was saved")


if __name__ == "__main__":
    main()
