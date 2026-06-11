# Word Error Rate (WER) Results

Comparison of WER across different modalities, detectors, and datasets.

| Modality | Detector | Dataset | WER (%) |
|----------|----------|---------|---------|
| Video | mediapipe | video_50 | 20.01% |
| Video | mediapipe | video_300 | 23.38% |
| Video | retinaface | video_50 | 17.06% |
| Video | retinaface | video_300 | 19.53% |
| Audio | none | video_50 | 1.96% |
| Audio | none | video_300 | 2.77% |
| Audiovisual | mediapipe | video_50 | 7.41% |
| Audiovisual | mediapipe | video_300 | 8.88% |
| Audiovisual | retinaface | video_50 | 7.11% |
| Audiovisual | retinaface | video_300 | 8.91% |

## Summary Statistics

### By Modality
- **Video**: 17.06% - 23.38% WER
- **Audio**: 1.96% - 2.77% WER (Best performance)
- **Audiovisual**: 7.11% - 8.91% WER

### By Detector (Video/AV only)
- **Retinaface**: Generally better for video-only; comparable for audiovisual
- **Mediapipe**: Slightly higher WER but faster processing

### By Dataset
- **video_50**: Generally lower WER across all modalities
- **video_300**: Slightly higher WER, possibly due to dataset characteristics
