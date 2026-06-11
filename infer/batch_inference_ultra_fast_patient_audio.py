#!/usr/bin/env python3
"""
Patient-specific audio-only inference pipeline.
Handles patient data naming convention (no repeat token).
"""
import os
import glob
import time
import json
import hydra
from omegaconf import DictConfig
import torch
import torchaudio
from datamodule.transforms import AudioTransform
from tqdm import tqdm
import gc


class UltraFastPatientAudioBatchInferencePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.modality = "audio"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} | Modality: audio (Patient)")
        self._init_model()

    def _init_model(self):
        self.audio_transform = AudioTransform(subset="test")

        # Load audio-only model
        from lightning import ModelModule

        ckpt = torch.load(self.cfg.pretrained_model_path, map_location=self.device)
        self.modelmodule = ModelModule(self.cfg)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule = self.modelmodule.to(self.device)
        self.modelmodule.eval()
        print(f"Audio model loaded on {self.device}")

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def process_single_audio(self, audio_path):
        try:
            start_time = time.time()

            # Load and process audio
            audio, sample_rate = self.load_audio(audio_path)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio).to(self.device, non_blocking=True)

            with torch.no_grad():
                transcript = self.modelmodule(audio)

            processing_time = time.time() - start_time

            # Cleanup
            del audio
            torch.cuda.empty_cache()

            return {
                "audio_path": audio_path,
                "transcript": transcript,
                "processing_time": processing_time,
                "status": "success",
            }
        except Exception as e:
            torch.cuda.empty_cache()
            return {
                "audio_path": audio_path,
                "transcript": None,
                "error": str(e),
                "status": "failed",
            }

    def process_batch(self, audio_paths, output_prefix, test_mode=False):
        if test_mode:
            audio_paths = audio_paths[:10]
            print(f"TEST MODE: Processing only {len(audio_paths)} audio files")

        print(f"Processing {len(audio_paths)} patient audio files (Audio-only)...")

        results = []
        successful = 0
        failed = 0
        total_processing_time = 0

        start_time = time.time()

        # Persistent output (JSONL + JSON snapshot)
        output_file = f"{output_prefix}.json"
        jsonl_file = f"{output_prefix}.jsonl"
        print(f"Output targets -> JSON: {os.path.abspath(output_file)} | JSONL: {os.path.abspath(jsonl_file)}")

        pbar = tqdm(audio_paths, desc="Processing patient audio files")
        for i, audio_path in enumerate(pbar):
            result = self.process_single_audio(audio_path)
            results.append(result)

            # Write to disk
            try:
                with open(jsonl_file, 'a', encoding='utf-8') as jf:
                    jf.write(json.dumps(result, ensure_ascii=False) + "\n")
                with open(output_file, 'w', encoding='utf-8') as sf:
                    json.dump(results, sf, indent=2, ensure_ascii=False)
            except Exception as io_err:
                print(f"Warning: failed to write results to disk: {io_err}", flush=True)

            if result["status"] == "success":
                successful += 1
                total_processing_time += result["processing_time"]
                fname = os.path.basename(result["audio_path"])
                print(f"✓ [{successful:3d}] {fname}: {result['transcript']}", flush=True)
                pbar.set_postfix({
                    "Success": successful,
                    "Failed": failed,
                    "GPU_Mem": f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
                })
            else:
                failed += 1
                fname = os.path.basename(result["audio_path"])
                print(f"✗ [{failed:3d}] {fname}: FAILED - {result['error']}", flush=True)
                pbar.set_postfix({
                    "Success": successful,
                    "Failed": failed,
                    "GPU_Mem": f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
                })

            # Periodic memory cleanup
            if (i + 1) % 3 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total_elapsed = time.time() - start_time
        print("\n=== Patient Audio Batch Processing Complete ===")
        print(f"Total audio files: {len(audio_paths)} | Success: {successful} | Failed: {failed}")
        print(f"Total elapsed time: {total_elapsed:.2f}s")
        if torch.cuda.is_available():
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        return results


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Get patient data directory
    patient_dir = "/scratch/th3482/LipVideoData/patient_25p"

    audio_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wav", "*.mp3", "*.flac"]
    audio_paths = []
    for ext in audio_extensions:
        audio_paths.extend(glob.glob(os.path.join(patient_dir, ext)))
    audio_paths.sort()

    print(f"Patient Dir={patient_dir} Found {len(audio_paths)} audio files")
    if len(audio_paths) == 0:
        print("No audio files found. Exiting.")
        return

    pipeline = UltraFastPatientAudioBatchInferencePipeline(cfg)

    ts = int(time.time())
    base = f"ultra_fast_patient_audio_{ts}"
    output_root = "/home/th3482/auto-avsr"
    os.makedirs(output_root, exist_ok=True)
    prefix = os.path.join(output_root, base)

    _ = pipeline.process_batch(audio_paths, prefix, test_mode=False)
    print(f"Completed Patient Audio. Outputs: {os.path.abspath(prefix)}.json/.jsonl")


if __name__ == "__main__":
    main()
