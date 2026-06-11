# Video 25fps Data - Processing Time Results

Comparison of average processing time per sample across different modalities and detectors on 25fps converted videos.

| Modality | Detector | Dataset | Avg Time (s) | Speedup vs Slowest |
|----------|----------|---------|--------------|-------------------|
| Video | mediapipe | video_50_25p | 2.701 | 4.29× |
| Video | mediapipe | video_300_25p | 2.401 | 4.83× |
| Video | retinaface | video_50_25p | 7.776 | 1.49× |
| Video | retinaface | video_300_25p | 7.697 | 1.51× |
| Audio | none | video_50_25p | 0.686 | 16.89× |
| Audio | none | video_300_25p | 0.703 | 16.48× |
| Audiovisual | mediapipe | video_50_25p | 2.371 | 4.89× |
| Audiovisual | mediapipe | video_300_25p | 1.743 | 6.65× |
| Audiovisual | retinaface | video_50_25p | 11.585 | 1.00× |
| Audiovisual | retinaface | video_300_25p | 8.971 | 1.29× |

## Summary Statistics

### By Modality
- **Audio**: 0.686 - 0.703s (Fastest, ~16-17× faster than slowest)
- **Video (mediapipe)**: 2.401 - 2.701s
- **Audiovisual (mediapipe)**: 1.743 - 2.371s
- **Video (retinaface)**: 7.697 - 7.776s
- **Audiovisual (retinaface)**: 8.971 - 11.585s (Slowest)

### By Detector
- **Mediapipe**: 1.743 - 2.701s (3-6× faster than retinaface)
- **Retinaface**: 7.697 - 11.585s (More accurate but slower)

### By Dataset
- **video_300_25p**: Generally faster than video_50_25p
- Consistent with original data pattern

## Comparison with Original FPS Data

| Modality/Detector | Original FPS Time | 25fps Time | Change |
|-------------------|-------------------|------------|--------|
| Audio (50) | 0.668s | 0.686s | +0.018s |
| Audio (300) | 0.696s | 0.703s | +0.007s |
| AV mediapipe (50) | 2.994s | 2.371s | **-0.623s** ✓ |
| AV mediapipe (300) | 2.274s | 1.743s | **-0.531s** ✓ |
| AV retinaface (50) | 8.880s | 11.585s | +2.705s |
| AV retinaface (300) | 11.612s | 8.971s | **-2.641s** ✓ |
| Video mediapipe (50) | 2.826s | 2.701s | -0.125s |
| Video mediapipe (300) | 2.122s | 2.401s | +0.279s |
| Video retinaface (50) | 8.186s | 7.776s | -0.410s |
| Video retinaface (300) | 10.406s | 7.697s | **-2.709s** ✓ |

### Key Observations

1. **Audio slightly slower**: Minimal impact from reading 25fps videos (~0.01-0.02s increase)

2. **Mediapipe AV faster**: Converting to 25fps speeds up mediapipe AV by ~0.5-0.6s
   - Better frame rate alignment reduces processing overhead
   - More efficient feature extraction at standard frame rate

3. **Retinaface mixed results**: 
   - 50fps data: slower after conversion (+2.7s for AV)
   - 300fps data: significantly faster after conversion (-2.6-2.7s)
   - Suggests retinaface detector benefits from consistent frame rate on variable-quality data

4. **Video retinaface improved**: Significant speedup (2-3s) for retinaface video processing on 300 dataset

## Throughput Estimates (25fps Data)

### Mediapipe
- Video: ~25-37 videos/minute
- Audiovisual: ~25-34 videos/minute

### Retinaface
- Video: ~8 videos/minute
- Audiovisual: ~5-7 videos/minute

### Audio-only
- ~85-87 audio files/minute

## Recommendations

1. **For best accuracy + speed**: Use 25fps with mediapipe AV (lowest WER, fast processing)
2. **For audio-only**: Frame rate conversion has minimal impact
3. **For retinaface users**: Convert to 25fps to improve speed on variable-quality datasets
4. **Overall**: **25fps conversion is beneficial for audiovisual tasks** - improves both accuracy and speed for mediapipe
