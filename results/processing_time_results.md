# Processing Time Results

Comparison of average processing time per sample across different modalities, detectors, and datasets.

| Modality | Detector | Dataset | Avg Time (s) | Speedup vs Slowest |
|----------|----------|---------|--------------|-------------------|
| Video | mediapipe | video_50 | 2.826 | 4.11× |
| Video | mediapipe | video_300 | 2.122 | 5.47× |
| Video | retinaface | video_50 | 8.186 | 1.42× |
| Video | retinaface | video_300 | 10.406 | 1.12× |
| Audio | none | video_50 | 0.668 | 17.38× |
| Audio | none | video_300 | 0.696 | 16.68× |
| Audiovisual | mediapipe | video_50 | 2.994 | 3.88× |
| Audiovisual | mediapipe | video_300 | 2.274 | 5.11× |
| Audiovisual | retinaface | video_50 | 8.880 | 1.31× |
| Audiovisual | retinaface | video_300 | 11.612 | 1.00× |

## Summary Statistics

### By Modality
- **Audio**: 0.668 - 0.696s (Fastest, ~17× faster than slowest)
- **Video (mediapipe)**: 2.122 - 2.826s
- **Audiovisual (mediapipe)**: 2.274 - 2.994s
- **Video (retinaface)**: 8.186 - 10.406s
- **Audiovisual (retinaface)**: 8.880 - 11.612s (Slowest)

### By Detector
- **Mediapipe**: 2.122 - 2.994s (3-5× faster than retinaface)
- **Retinaface**: 8.186 - 11.612s (More accurate but slower)

### By Dataset
- **video_300**: Generally faster than video_50 for same modality/detector
- This suggests video_300 may have shorter average clip duration

## Throughput Estimates

### Mediapipe
- Video: ~28-34 videos/minute
- Audiovisual: ~24-26 videos/minute

### Retinaface
- Video: ~6-7 videos/minute
- Audiovisual: ~5-7 videos/minute

### Audio-only
- ~86-90 audio files/minute (fastest option)
