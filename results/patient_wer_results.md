# Patient Data - Word Error Rate (WER) Results

Comparison of WER across different modalities and detectors on patient data.

| Modality | Detector | WER (%) |
|----------|----------|---------|
| Video | mediapipe | 346.48% |
| Video | retinaface | 467.94% |
| Audio | none | 85.14% |
| Audiovisual | mediapipe | 87.27% |
| Audiovisual | retinaface | 95.72% |

## Summary Statistics

### By Modality
- **Audio**: 85.14% WER (Best performance)
- **Audiovisual**: 87.27% - 95.72% WER
- **Video**: 346.48% - 467.94% WER (Poor performance)

### By Detector (Video/AV only)
- **Mediapipe**: Generally better performance than Retinaface on patient data
  - Video: 346.48% vs 467.94%
  - Audiovisual: 87.27% vs 95.72%
- **Retinaface**: Higher WER across all modalities on patient data

## Key Observations

1. **High WER for Video-only**: The extremely high WER (>300%) for video-only modalities suggests:
   - Patient speech patterns may be significantly different from training data
   - Lip movements may be less distinct or unusual in patient population
   - Visual-only recognition is highly challenging for this specific dataset

2. **Audio performs best**: Audio-only recognition achieves the lowest WER, indicating that acoustic information is more reliable than visual for this patient cohort.

3. **Audiovisual helps but limited**: While AV reduces WER compared to video-only, it doesn't match audio-only performance, suggesting visual modality may introduce noise rather than useful information for this dataset.

4. **Model generalization challenge**: The dramatic performance drop compared to standard datasets (video_50/300) indicates limited model generalization to patient-specific speech characteristics.
