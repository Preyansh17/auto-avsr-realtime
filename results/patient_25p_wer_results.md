# Patient 25fps Data - Word Error Rate (WER) Results

Comparison of WER across different modalities and detectors on patient data converted to 25fps.

| Modality | Detector | WER (%) |
|----------|----------|---------|
| Video | mediapipe | 128.82% |
| Video | retinaface | 150.17% |
| Audio | none | 85.60% |
| Audiovisual | mediapipe | 90.99% |
| Audiovisual | retinaface | 88.50% |

## Summary Statistics

### By Modality
- **Audio**: 85.60% WER (Best performance)
- **Audiovisual**: 88.50% - 90.99% WER
- **Video**: 128.82% - 150.17% WER (Poor performance)

### By Detector (Video/AV only)
- **Mediapipe**: 
  - Video: 128.82%
  - Audiovisual: 90.99%
- **Retinaface**: 
  - Video: 150.17%
  - Audiovisual: 88.50% (Better for AV)

## Comparison with Original Patient Data (Original FPS)

| Modality | Original FPS WER | 25fps WER | Change |
|----------|------------------|-----------|--------|
| Audio | 85.14% | 85.60% | +0.46% |
| AV mediapipe | 87.27% | 90.99% | +3.72% |
| AV retinaface | 95.72% | 88.50% | **-7.22%** ✓ |
| Video mediapipe | 346.48% | 128.82% | **-217.66%** ✓✓ |
| Video retinaface | 467.94% | 150.17% | **-317.77%** ✓✓ |

### Key Observations

1. **Dramatic improvement for video-only**: 25fps conversion reduces video-only WER by 200-300%
   - Still high (>100%), but much more usable than original
   - Suggests original variable frame rate was a major issue for visual processing

2. **Audio unchanged**: Frame rate has minimal impact (~0.5% change)

3. **Mixed results for audiovisual**:
   - Mediapipe AV: slightly worse (+3.7%)
   - Retinaface AV: significantly better (-7.2%) ✓

4. **Retinaface benefits more**: Frame standardization particularly helps retinaface detector on patient data

5. **Still challenging dataset**: Even with 25fps, patient data remains difficult:
   - Video WER still >100% (predicting more words than actual)
   - Audio/AV WER ~86-91% (vs 2-5% on standard datasets)
   - Indicates fundamental domain shift, not just frame rate issue

## Recommendations for Patient Data

1. **Use audio-only** for production systems (85.60% WER, fastest)
2. **If using AV, prefer retinaface + 25fps** (88.50% WER, best AV performance)
3. **Video-only still not viable** even at 25fps (>128% WER)
4. **Consider domain adaptation**: Fine-tune models on patient-specific data
5. **Collect more patient data**: Current models poorly generalize to this population
