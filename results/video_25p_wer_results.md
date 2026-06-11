# Video 25fps Data - Word Error Rate (WER) Results

Comparison of WER across different modalities and detectors on 25fps converted videos.

| Modality | Detector | Dataset | WER (%) |
|----------|----------|---------|---------|
| Video | mediapipe | video_50_25p | 21.17% |
| Video | mediapipe | video_300_25p | 26.21% |
| Video | retinaface | video_50_25p | 17.52% |
| Video | retinaface | video_300_25p | 21.84% |
| Audio | none | video_50_25p | 1.99% |
| Audio | none | video_300_25p | 2.77% |
| Audiovisual | mediapipe | video_50_25p | 3.12% |
| Audiovisual | mediapipe | video_300_25p | 5.12% |
| Audiovisual | retinaface | video_50_25p | 3.50% |
| Audiovisual | retinaface | video_300_25p | 5.08% |

## Summary Statistics

### By Modality
- **Video**: 17.52% - 26.21% WER
- **Audio**: 1.99% - 2.77% WER (Best performance)
- **Audiovisual**: 3.12% - 5.12% WER

### By Detector (Video/AV only)
- **Retinaface**: Generally better for video-only; comparable for audiovisual
- **Mediapipe**: Slightly higher WER but faster processing

### By Dataset
- **video_50_25p**: Generally lower WER across all modalities
- **video_300_25p**: Slightly higher WER, consistent with original data

## Comparison with Original FPS Data

| Modality | Original FPS WER | 25fps WER | Change |
|----------|------------------|-----------|--------|
| Audio (50) | 1.96% | 1.99% | +0.03% |
| Audio (300) | 2.77% | 2.77% | 0% |
| AV mediapipe (50) | 7.41% | 3.12% | **-4.29%** ✓ |
| AV mediapipe (300) | 8.88% | 5.12% | **-3.76%** ✓ |
| Video mediapipe (50) | 20.01% | 21.17% | +1.16% |
| Video mediapipe (300) | 23.38% | 26.21% | +2.83% |

### Key Observations

1. **Audio unchanged**: Frame rate has no impact on audio-only performance (as expected)

2. **Audiovisual significantly improved**: Converting to 25fps improves AV performance by 3-4% WER, suggesting:
   - Better temporal alignment with training data
   - More stable visual features at consistent frame rate
   - Reduced frame interpolation artifacts

3. **Video-only slightly worse**: Small degradation in video-only performance may be due to:
   - Information loss during frame rate conversion
   - Training data frame rate mismatch

4. **Overall recommendation**: **Use 25fps for audiovisual tasks** - provides best accuracy improvement
