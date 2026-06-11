#!/usr/bin/env python3
"""
Test script to verify training configuration before submitting job.
"""
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def test_config(cfg: DictConfig):
    print("=" * 60)
    print("Testing Training Configuration")
    print("=" * 60)
    
    # Override with training parameters
    cfg.data.modality = 'video'
    cfg.data.dataset = OmegaConf.create({
        'root_dir': '/scratch/th3482/LipVideoData/video_50_25p_crops',
        'label_dir': 'labels',
        'train_file': 'train.csv',
        'val_file': 'val.csv',
        'test_file': 'test.csv'
    })
    cfg.exp_dir = '/scratch/th3482/auto-avsr/experiments'
    cfg.exp_name = 'test_config'
    cfg.pretrained_model_path = '/home/th3482/auto-avsr/cpts/vsr_trlrwlrs2lrs3vox2avsp_base.pth'
    cfg.vocab_file = '/home/th3482/auto-avsr/sentence.txt'
    cfg.beam_size = 40
    cfg.ctc_weight = 0.1
    cfg.pre_beam_ratio = 1.5
    
    print("\n[Config Summary]")
    print(f"  Modality: {cfg.data.modality}")
    print(f"  Dataset root: {cfg.data.dataset.root_dir}")
    print(f"  Train file: {cfg.data.dataset.train_file}")
    print(f"  Val file: {cfg.data.dataset.val_file}")
    print(f"  Pretrained model: {cfg.pretrained_model_path}")
    print(f"  Vocab file: {cfg.vocab_file}")
    print(f"  Beam size: {cfg.beam_size}")
    print(f"  CTC weight: {cfg.ctc_weight}")
    print(f"  Pre-beam ratio: {cfg.pre_beam_ratio}")
    
    # Check if files exist
    print("\n[File Verification]")
    checks = [
        ("Dataset root", cfg.data.dataset.root_dir),
        ("Train CSV", os.path.join(cfg.data.dataset.root_dir, cfg.data.dataset.label_dir, cfg.data.dataset.train_file)),
        ("Val CSV", os.path.join(cfg.data.dataset.root_dir, cfg.data.dataset.label_dir, cfg.data.dataset.val_file)),
        ("Test CSV", os.path.join(cfg.data.dataset.root_dir, cfg.data.dataset.label_dir, cfg.data.dataset.test_file)),
        ("Pretrained model", cfg.pretrained_model_path),
        ("Vocab file", cfg.vocab_file),
    ]
    
    all_ok = True
    for name, path in checks:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    # Try to load vocab file
    if os.path.exists(cfg.vocab_file):
        print("\n[Vocabulary File Preview]")
        with open(cfg.vocab_file, 'r') as f:
            lines = f.readlines()
            print(f"  Total sentences: {len(lines)}")
            print(f"  First 5 sentences:")
            for i, line in enumerate(lines[:5]):
                print(f"    {i+1}. {line.strip()}")
    
    # Check train CSV format
    train_csv = os.path.join(cfg.data.dataset.root_dir, cfg.data.dataset.label_dir, cfg.data.dataset.train_file)
    if os.path.exists(train_csv):
        print("\n[Training Data Preview]")
        with open(train_csv, 'r') as f:
            lines = f.readlines()
            print(f"  Total training samples: {len(lines)}")
            print(f"  First 3 samples:")
            for i, line in enumerate(lines[:3]):
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    print(f"    {i+1}. {parts[1].split('/')[-1][:50]}...")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All checks passed! Ready to train.")
        print("  Submit job: sbatch train_video50_vocab.sbatch")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(test_config())


