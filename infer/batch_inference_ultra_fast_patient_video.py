#!/usr/bin/env python3
import os
import sys
import glob
import time
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torchaudio
import torchvision
import wandb
from typing import List

# Add parent directory to path to import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datamodule.transforms import AudioTransform, VideoTransform
from datamodule.av_dataset import cut_or_pad
from tqdm import tqdm
import gc
import re


def normalize_text(text: str) -> List[str]:
    if text is None:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text(reference)
    hyp_tokens = normalize_text(hypothesis)
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    dist = levenshtein_distance(ref_tokens, hyp_tokens)
    return dist / float(len(ref_tokens))


def extract_groundtruth_from_filename(video_path: str) -> str:
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    # ground truth is the first segment before the first underscore
    # e.g., "It is about the exposure in this case_47-50_repeat5_..."
    gt = name.split("_")[0]
    # Replace dashes with spaces for cleaner tokenization
    gt = gt.replace("-", " ")
    return gt


class UltraFastBatchInferencePipeline:
    def __init__(self, cfg, detector="retinaface", vocab_file=None):
        self.cfg = cfg
        self.modality = cfg.data.modality
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = detector
        self.vocab_file = vocab_file
        self.allowed_token_ids = None
        
        print(f"Using device: {self.device}")
        
        # 加载词汇限制（如果指定）
        if self.vocab_file:
            self._load_vocab_constraint()
        
        # 初始化模型和相关组件
        self._init_model()
    
    def _load_vocab_constraint(self):
        """从句子文件中加载词汇限制并转换为token IDs"""
        from preparation.transforms import TextTransform
        
        print(f"\n{'='*70}")
        print(f"Loading vocabulary constraint from sentences: {self.vocab_file}")
        print(f"{'='*70}")
        
        # 读取句子文件并处理
        sentences = []
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip().upper().replace(" ", " ")
                if sentence:  # 跳过空行
                    sentences.append(sentence)
        
        print(f"Loaded {len(sentences)} sentences from vocabulary file")
        
        # 初始化TextTransform
        text_transform = TextTransform()
        
        # 收集所有唯一的token IDs
        allowed_tokens = set()
        word_to_tokens = {}
        
        for sentence in sentences:
            # 将句子转换为token IDs
            token_ids = text_transform.tokenize(sentence)
            token_list = [_.item() for _ in token_ids]
            allowed_tokens.update(token_list)
            
            # 记录每个词的token映射（用于调试）
            words = sentence.split()
            if len(words) == 1:  # 单个词
                word_to_tokens[sentence] = token_list
            else:  # 多个词，记录整个句子
                word_to_tokens[sentence] = token_list
        
        # 添加blank token (ID=0) - CTC需要
        allowed_tokens.add(0)
        
        # 转换为列表并排序
        self.allowed_token_ids = sorted(list(allowed_tokens))
        
        print(f"\nVocabulary constraint loaded:")
        print(f"  - Total sentences: {len(sentences)}")
        print(f"  - Total unique tokens: {len(self.allowed_token_ids)}")
        print(f"  - Allowed token IDs (sorted): {self.allowed_token_ids}")
        
        # 打印一些示例映射
        print(f"\nSample word-to-token mappings:")
        for i, (word, tokens) in enumerate(list(word_to_tokens.items())[:10]):
            print(f"  {word} → {tokens}")
        if len(word_to_tokens) > 10:
            print(f"  ... and {len(word_to_tokens) - 10} more")
        
        print(f"{'='*70}\n")
    
    def _init_model(self):
        """初始化模型和相关组件"""
        if self.modality in ["video", "audiovisual"]:
            if self.detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device=str(self.device))
                self.video_process = VideoProcess(convert_gray=False)
            elif self.detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            else:
                raise ValueError(f"Unknown detector: {self.detector}")
            self.video_transform = VideoTransform(subset="test")

        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")

        # 加载模型
        if self.modality == "video":
            from lightning import ModelModule
        elif self.modality == "audio":
            from lightning_av import ModelModule
        elif self.modality == "audiovisual":
            from lightning_av import ModelModule
        else:
            raise ValueError(f"Unknown modality: {self.modality}")
            
        ckpt = torch.load(self.cfg.pretrained_model_path, map_location=self.device)
        self.modelmodule = ModelModule(self.cfg)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule = self.modelmodule.to(self.device)
        self.modelmodule.eval()
        
        print(f"Model loaded on {self.device}")


    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def process_single_video(self, video_path):
        """处理单个视频文件 - 超优化版本"""
        try:
            start_time = time.time()
            
            # 加载和处理视频
            if self.modality in ["video", "audiovisual"]:
                video = self.load_video(video_path)
                landmarks = self.landmarks_detector(video)
                video = self.video_process(video, landmarks)
                video = torch.tensor(video)
                video = video.permute((0, 3, 1, 2))
                video = self.video_transform(video)
                # 立即移动到GPU，使用non_blocking=True
                video = video.to(self.device, non_blocking=True)

            # 加载和处理音频
            if self.modality in ["audio", "audiovisual"]:
                audio, sample_rate = self.load_audio(video_path)
                audio = self.audio_process(audio, sample_rate)
                audio = audio.transpose(1, 0)
                audio = self.audio_transform(audio)
                # 立即移动到GPU，使用non_blocking=True
                audio = audio.to(self.device, non_blocking=True)

            # 推理
            with torch.no_grad():
                if self.modality == "video":
                    transcript, token_ids = self.modelmodule(video, allowed_token_ids=self.allowed_token_ids, return_tokens=True)
                elif self.modality == "audio":
                    transcript, token_ids = self.modelmodule(audio, allowed_token_ids=self.allowed_token_ids, return_tokens=True)
                elif self.modality == "audiovisual":
                    transcript, token_ids = self.modelmodule(video, audio, allowed_token_ids=self.allowed_token_ids, return_tokens=True)

            processing_time = time.time() - start_time
            
            # 立即清理GPU内存
            if self.modality in ["video", "audiovisual"]:
                del video
            if self.modality in ["audio", "audiovisual"]:
                del audio
            torch.cuda.empty_cache()
            
            # 计算WER
            ground_truth = extract_groundtruth_from_filename(video_path)
            wer = compute_wer(ground_truth, str(transcript))
            
            # 打印token信息用于调试
            if token_ids is not None:
                # 将token IDs转换为文本用于显示
                from preparation.transforms import TextTransform
                text_transform = TextTransform()
                token_texts = []
                for token_id in token_ids:
                    if token_id == 0:  # blank token
                        token_texts.append('<blank>')
                    elif token_id == 5047:  # EOS token
                        token_texts.append('<eos>')
                    else:
                        # 从token ID反向查找对应的文本
                        try:
                            # 这里需要根据实际的token映射来实现
                            token_texts.append(f'<{token_id}>')
                        except:
                            token_texts.append(f'<{token_id}>')
                
                print(f"Token IDs: {token_ids}")
                print(f"Tokens: {token_texts}")
            
            return {
                "video_path": video_path,
                "transcript": transcript,
                "ground_truth": ground_truth,
                "wer": wer,
                "token_ids": token_ids if token_ids is not None else None,
                "processing_time": processing_time,
                "status": "success"
            }
            
        except Exception as e:
            torch.cuda.empty_cache()
            return {
                "video_path": video_path,
                "transcript": None,
                "ground_truth": None,
                "wer": None,
                "token_ids": None,
                "error": str(e),
                "status": "failed"
            }

    def load_audio(self, video_path):
        """从视频文件中提取音频"""
        try:
            # 使用torchvision读取视频并提取音频
            video_data = torchvision.io.read_video(video_path, pts_unit="sec")
            video_tensor = video_data[0]  # video frames
            audio_tensor = video_data[1]  # audio tensor
            sample_rate = video_data[2]   # sample rate
            
            if audio_tensor is None:
                raise ValueError(f"No audio found in video: {video_path}")
            
            return audio_tensor, sample_rate
        except Exception as e:
            raise ValueError(f"Failed to load audio from {video_path}: {str(e)}")

    def process_batch_ultra_fast(self, video_paths, output_file=None, test_mode=False):
        """超快速批量处理"""
        if test_mode:
            video_paths = video_paths[:10]  # 测试模式只处理前10个
            print(f"TEST MODE: Processing only {len(video_paths)} videos")
        
        print(f"Processing {len(video_paths)} videos with ultra-fast optimization...")
        
        # Initialize wandb run
        run_name = f"patient_inference_{int(time.time())}"
        wandb.init(
            project="patient_vocab_comparison",
            name=run_name,
            config={
                "modality": self.modality,
                "detector": self.detector,
                "beam_size": self.cfg.get("beam_size", "default"),
                "pre_beam_ratio": self.cfg.get("pre_beam_ratio", "default"),
                "ctc_weight": self.cfg.get("ctc_weight", "default"),
                "video_dirs": getattr(self.cfg, "video_dirs", "patient_25p"),
                "vocab_constraint": self.allowed_token_ids is not None,
                "total_videos": len(video_paths)
            },
            # mode="offline"  # SSL certificate issue resolved
        )
        
        results = []
        successful = 0
        failed = 0
        total_processing_time = 0
        
        start_time = time.time()

        # 文件输出设置：逐条写入JSONL，持续快照JSON
        if output_file is None:
            output_file = f"ultra_fast_results_{int(start_time)}.json"
        jsonl_file = output_file[:-5] + ".jsonl" if output_file.endswith('.json') else output_file + ".jsonl"
        print(f"Output targets -> JSON: {os.path.abspath(output_file)} | JSONL: {os.path.abspath(jsonl_file)}")
        
        # 使用进度条
        pbar = tqdm(video_paths, desc="Processing videos")
        
        for i, video_path in enumerate(pbar):
            result = self.process_single_video(video_path)
            results.append(result)

            # 立即写入结果到JSONL和更新JSON快照
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
                filename = os.path.basename(result["video_path"])
                wer = result.get("wer", 0.0)
                if result.get("token_ids") is not None:
                    print(f"✓ [{successful:3d}] {filename}: {result['transcript']} (WER: {wer:.3f})", flush=True)
                else:
                    print(f"✓ [{successful:3d}] {filename}: {result['transcript']} (WER: {wer:.3f})", flush=True)
                pbar.set_postfix({
                    "Success": successful,
                    "Failed": failed,
                    "Current": os.path.basename(video_path)[:25] + "...",
                    "Transcript": str(result["transcript"])[:25] + "...",
                    "WER": f"{wer:.3f}",
                    "GPU_Mem": f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
                    "Avg_Time": f"{total_processing_time/successful:.1f}s" if successful > 0 else "N/A",
                    "ETA": f"{(len(video_paths)-i-1)*total_processing_time/successful/60:.1f}min" if successful > 0 else "N/A"
                })
            else:
                failed += 1
                # 立即输出错误
                filename = os.path.basename(result["video_path"])
                print(f"✗ [{failed:3d}] {filename}: FAILED - {result['error']}", flush=True)
                pbar.set_postfix({
                    "Success": successful,
                    "Failed": failed,
                    "Error": str(result["error"])[:25] + "...",
                    "GPU_Mem": f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
                })
            
            # 每3个视频清理一次内存（更频繁的清理）
            if (i + 1) % 3 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        total_elapsed = time.time() - start_time
        
        # 计算总体WER统计
        total_wer = 0.0
        wer_count = 0
        for result in results:
            if result.get("status") == "success" and result.get("wer") is not None:
                total_wer += result["wer"]
                wer_count += 1
        
        avg_wer = total_wer / wer_count if wer_count > 0 else 0.0
        
        # 打印数据和参数信息
        print(f"\n{'='*80}")
        print(f"INFERENCE PARAMETERS AND DATA SUMMARY")
        print(f"{'='*80}")
        print(f"Model Configuration:")
        print(f"  - Modality: {self.modality}")
        print(f"  - Detector: {self.detector}")
        print(f"  - Device: {self.device}")
        print(f"  - Vocabulary constraint: {'Enabled' if self.allowed_token_ids is not None else 'Disabled'}")
        if self.allowed_token_ids is not None:
            print(f"  - Allowed tokens: {len(self.allowed_token_ids)}")
            print(f"  - Allowed token IDs: {sorted(self.allowed_token_ids)}")
        
        print(f"\nBeam Search Parameters:")
        print(f"  - Beam size: {self.cfg.get('beam_size', 'default')}")
        print(f"  - Pre-beam ratio: {self.cfg.get('pre_beam_ratio', 'default')}")
        print(f"  - CTC weight: {self.cfg.get('ctc_weight', 'default')}")
        
        print(f"\nDataset Information:")
        print(f"  - Total videos processed: {len(video_paths)}")
        print(f"  - Successful: {successful}")
        print(f"  - Failed: {failed}")
        print(f"  - Success rate: {successful/len(video_paths)*100:.1f}%")
        
        print(f"\nSample Results (first 5 successful):")
        sample_count = 0
        for result in results:
            if result.get("status") == "success" and sample_count < 5:
                filename = os.path.basename(result["video_path"])
                gt = result.get("ground_truth", "N/A")
                pred = result.get("transcript", "N/A")
                wer = result.get("wer", 0.0)
                print(f"  [{sample_count+1}] {filename}")
                print(f"      Ground Truth: {gt}")
                print(f"      Prediction:   {pred}")
                print(f"      WER:         {wer:.3f}")
                sample_count += 1
        
        print(f"\n{'='*80}")
        print(f"FINAL WER STATISTICS")
        print(f"{'='*80}")
        print(f"Average WER: {avg_wer:.3f}")
        print(f"Total processing time: {total_processing_time:.2f}s")
        print(f"Total elapsed time: {total_elapsed:.2f}s")
        print(f"Average time per video: {total_processing_time/successful:.2f}s" if successful > 0 else "N/A")
        print(f"Throughput: {successful/total_elapsed:.2f} videos/second")
        print(f"Estimated time for 300 videos: {300*total_elapsed/len(video_paths)/60:.1f} minutes")
        
        if torch.cuda.is_available():
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        
        # Log metrics to wandb
        wandb.log({
            "avg_wer": avg_wer,
            "success_rate": successful/len(video_paths),
            "total_processing_time": total_processing_time,
            "total_elapsed_time": total_elapsed,
            "throughput": successful/total_elapsed,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0
        })
        
        # Log sample results to wandb
        sample_results = []
        for result in results:  # Log all successful results
            if result.get("status") == "success":
                sample_results.append({
                    "video_path": os.path.basename(result["video_path"]),
                    "ground_truth": result.get("ground_truth", ""),
                    "prediction": str(result.get("transcript", "")),
                    "wer": result.get("wer", 0.0)
                })
        
        wandb.log({"sample_results": wandb.Table(
            columns=["video_path", "ground_truth", "prediction", "wer"],
            data=[[r["video_path"], r["ground_truth"], r["prediction"], r["wer"]] for r in sample_results]
        )})
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")
        
        wandb.finish()
        return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 可调参数：detector(s) 与 video_dirs，支持逗号分隔
    
    # 从命令行参数获取detector和video_dirs
    detectors = getattr(cfg, "detectors", "mediapipe").split(",")
    video_dirs = getattr(cfg, "video_dirs", "/scratch/th3482/LipVideoData/patient_25p").split(",")
    vocab_file = getattr(cfg, "vocab_file", None)
    
    # 清理路径
    detectors = [d.strip() for d in detectors]
    video_dirs = [d.strip() for d in video_dirs]
    
    print(f"Detectors: {detectors}")
    print(f"Video directories: {video_dirs}")
    print(f"Vocabulary file: {vocab_file}")


    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]

    for detector in detectors:
        for video_dir in video_dirs:
            video_paths = []
            for ext in video_extensions:
                video_paths.extend(glob.glob(os.path.join(video_dir, ext)))
            video_paths.sort()

            print(f"Detector={detector} Dir={video_dir} Found {len(video_paths)} video files")
            if len(video_paths) == 0:
                continue

            pipeline = UltraFastBatchInferencePipeline(cfg, detector=detector, vocab_file=vocab_file)

            dir_tag = os.path.basename(video_dir.rstrip('/'))
            timestamp = int(time.time())
            
            # Add beam search parameters to filename
            beam_size = cfg.get("beam_size", 40)
            pre_beam_ratio = cfg.get("pre_beam_ratio", 1.5)
            ctc_weight = cfg.get("ctc_weight", 0.1)
            base_name = f"ultra_fast_{dir_tag}_{detector}_beam{beam_size}_pbr{pre_beam_ratio}_ctc{ctc_weight}_{timestamp}.json"
            
            output_root = "/home/th3482/auto-avsr"
            os.makedirs(output_root, exist_ok=True)
            output_file = os.path.join(output_root, base_name)

            _ = pipeline.process_batch_ultra_fast(video_paths, output_file, test_mode=False)

            print(f"Completed Detector={detector} Dir={video_dir}. Results: {os.path.abspath(output_file)} and JSONL sidecar.")


if __name__ == "__main__":
    main()