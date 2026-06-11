# Patient Data - Processing Time Results

Comparison of average processing time per sample across different modalities and detectors on patient data.

| Modality | Detector | Avg Time (s) | Speedup vs Slowest |
|----------|----------|--------------|-------------------|
| Video | mediapipe | 4.403 | 3.40× |
| Video | retinaface | 10.641 | 1.41× |
| Audio | none | 0.798 | 18.78× |
| Audiovisual | mediapipe | 3.278 | 4.57× |
| Audiovisual | retinaface | 14.990 | 1.00× |

## Summary Statistics

### By Modality
- **Audio**: 0.798s (Fastest, ~19× faster than slowest)
- **Audiovisual (mediapipe)**: 3.278s
- **Video (mediapipe)**: 4.403s
- **Video (retinaface)**: 10.641s
- **Audiovisual (retinaface)**: 14.990s (Slowest)

### By Detector
- **Mediapipe**: 3.278 - 4.403s (3-4.5× faster than retinaface)
- **Retinaface**: 10.641 - 14.990s (Slower but more robust detection)

### Comparison with Standard Datasets

Patient data shows similar processing time patterns to video_50/300:
- Audio remains fastest (~0.8s)
- Mediapipe is 3-4× faster than Retinaface
- Audiovisual slightly slower than video-only for same detector

**Note**: Slightly longer processing times for patient data may be due to:
- Longer average video duration
- More challenging facial detection scenarios
- Variable video quality

## Throughput Estimates

### Mediapipe
- Video: ~14 videos/minute
- Audiovisual: ~18 videos/minute

### Retinaface
- Video: ~6 videos/minute
- Audiovisual: ~4 videos/minute

### Audio-only
- ~75 audio files/minute (fastest option)

## Recommendations

1. **For real-time patient monitoring**: Use audio-only (fastest, best accuracy)
2. **For research/evaluation**: Use audiovisual with mediapipe (good speed-accuracy tradeoff)
3. **For maximum robustness**: Use retinaface if detection quality is critical (despite slower speed)
