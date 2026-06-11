import os
import hydra
import logging
from omegaconf import OmegaConf

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from avg_ckpts import ensemble
from datamodule.data_module import DataModule


@hydra.main(version_base="1.3", config_path="configs", config_name="train_config")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Ensure dataset config is structured if a string key like 'video_50' was passed
    if isinstance(getattr(cfg.data, "dataset", None), str):
        ds_name = cfg.data.dataset
        ds_path = os.path.join("configs", "data", "dataset", f"{ds_name}.yaml")
        if os.path.exists(ds_path):
            cfg.data.dataset = OmegaConf.load(ds_path)
        else:
            raise FileNotFoundError(f"Dataset config not found: {ds_path}")

    # Set modules and trainer
    # Accept both "video" and legacy "visual" for visual-only modality
    if cfg.data.modality in ["audio", "video", "visual"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av import ModelModule
    else:
        raise ValueError(f"Unsupported modality: {cfg.data.modality}")
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    
    # Determine wandb project name based on whether vocab constraint is used
    wandb_project = "video50_vocab_training" if hasattr(cfg, 'vocab_file') and cfg.vocab_file else "auto_avsr"
    
    # Create wandb logger with comprehensive config
    # Resolve dataset field for logging
    dataset_field = getattr(cfg.data, 'dataset', None)
    if isinstance(dataset_field, str):
        dataset_display = dataset_field
    elif hasattr(dataset_field, 'root_dir'):
        dataset_display = dataset_field.root_dir
    elif isinstance(dataset_field, dict) and 'root_dir' in dataset_field:
        dataset_display = dataset_field['root_dir']
    else:
        dataset_display = 'unknown'

    wandb_logger = WandbLogger(
        name=cfg.exp_name, 
        project=wandb_project,
        config={
            # Data configuration
            "dataset": dataset_display,
            "modality": cfg.data.modality,
            "max_frames": cfg.data.max_frames,
            "max_frames_val": cfg.data.max_frames_val,
            
            # Model configuration
            "model_architecture": "ResNet+Conformer",
            "encoder_layers": cfg.model.visual_backbone.elayers if cfg.data.modality == "video" else cfg.model.audio_backbone.elayers,
            "decoder_layers": cfg.model.visual_backbone.dlayers if cfg.data.modality == "video" else cfg.model.audio_backbone.dlayers,
            "attention_dim": cfg.model.visual_backbone.adim if cfg.data.modality == "video" else cfg.model.audio_backbone.adim,
            "attention_heads": cfg.model.visual_backbone.aheads if cfg.data.modality == "video" else cfg.model.audio_backbone.aheads,
            
            # Training configuration
            "max_epochs": cfg.trainer.max_epochs,
            "gpus": cfg.gpus,
            "num_nodes": cfg.trainer.num_nodes,
            "precision": cfg.trainer.precision,
            "gradient_clip_val": cfg.trainer.gradient_clip_val,
            "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
            
            # Optimizer configuration
            "optimizer": cfg.optimizer.name,
            "learning_rate": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
            "warmup_epochs": cfg.optimizer.warmup_epochs,
            
            # Loss configuration
            "mtlalpha": cfg.model.visual_backbone.mtlalpha if cfg.data.modality == "video" else cfg.model.audio_backbone.mtlalpha,
            "lsm_weight": cfg.model.visual_backbone.lsm_weight if cfg.data.modality == "video" else cfg.model.audio_backbone.lsm_weight,
            
            # Beam search configuration (if specified)
            "beam_size": getattr(cfg, 'beam_size', None),
            "ctc_weight": getattr(cfg, 'ctc_weight', None),
            "pre_beam_ratio": getattr(cfg, 'pre_beam_ratio', None),
            
            # Vocabulary constraint
            "vocab_file": getattr(cfg, 'vocab_file', None),
            "vocab_constrained": hasattr(cfg, 'vocab_file') and cfg.vocab_file is not None,
            
            # Pretrained model
            "pretrained_model_path": cfg.pretrained_model_path if cfg.pretrained_model_path else None,
            "transfer_frontend": cfg.transfer_frontend if cfg.transfer_frontend else False,
            "transfer_encoder": cfg.transfer_encoder if cfg.transfer_encoder else False,

            # LoRA configuration (if present)
            "lora_enabled": getattr(getattr(cfg, 'lora', {}), 'enabled', False),
            "lora_r": getattr(getattr(cfg, 'lora', {}), 'r', None),
            "lora_alpha": getattr(getattr(cfg, 'lora', {}), 'alpha', None),
            "lora_dropout": getattr(getattr(cfg, 'lora', {}), 'dropout', None),
            "lora_scopes": getattr(getattr(cfg, 'lora', {}), 'scopes', None),
            "lora_name_patterns": getattr(getattr(cfg, 'lora', {}), 'name_patterns', None),
            
            # Freeze configuration (if present)
            "freeze_enabled": getattr(getattr(cfg, 'freeze', {}), 'enabled', False),
            "freeze_scopes": getattr(getattr(cfg, 'freeze', {}), 'scopes', None),
            "freeze_name_patterns": getattr(getattr(cfg, 'freeze', {}), 'name_patterns', None),
        }
    )
    
    # Log vocab file content if specified
    if hasattr(cfg, 'vocab_file') and cfg.vocab_file and os.path.exists(cfg.vocab_file):
        with open(cfg.vocab_file, 'r') as f:
            vocab_sentences = [line.strip() for line in f.readlines()]
        wandb_logger.experiment.config.update({
            "vocab_size": len(vocab_sentences),
            "vocab_preview": vocab_sentences[:10]  # First 10 sentences
        })
    
    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)
    ensemble(cfg)


if __name__ == "__main__":
    main()
