# Patient 25fps Data - Processing Time Results

Comparison of average processing time per sample across different modalities and detectors on patient data converted to 25fps.

| Modality | Detector | Avg Time (s) | Speedup vs Slowest |
|----------|----------|--------------|-------------------|
| Video | mediapipe | 3.162 | 4.58× |
| Video | retinaface | 14.495 | 1.00× |
| Audio | none | 0.756 | 19.17× |
| Audiovisual | mediapipe | 3.175 | 4.57× |
| Audiovisual | retinaface | 13.906 | 1.04× |

## Summary Statistics

### By Modality
- **Audio**: 0.756s (Fastest, ~19× faster than slowest)
- **Video (mediapipe)**: 3.162s
- **Audiovisual (mediapipe)**: 3.175s
- **Audiovisual (retinaface)**: 13.906s
- **Video (retinaface)**: 14.495s (Slowest)

### By Detector
- **Mediapipe**: 3.162 - 3.175s (4.5× faster than retinaface)
- **Retinaface**: 13.906 - 14.495s (Slower but more robust)

## Comparison with Original Patient Data (Original FPS)

| Modality/Detector | Original FPS Time | 25fps Time | Change |
|-------------------|-------------------|------------|--------|
| Audio | 0.798s | 0.756s | **-0.042s** ✓ |
| AV mediapipe | 3.278s | 3.175s | **-0.103s** ✓ |
| AV retinaface | 14.990s | 13.906s | **-1.084s** ✓ |
| Video mediapipe | 4.403s | 3.162s | **-1.241s** ✓ |
| Video retinaface | 10.641s | 14.495s | +3.854s |

### Key Observations

1. **Most modalities faster**: 25fps conversion speeds up 4 out of 5 configurations
   - Audio: 5% faster (-0.04s)
   - AV mediapipe: 3% faster (-0.1s)
   - AV retinaface: 7% faster (-1.1s)
   - Video mediapipe: 28% faster (-1.2s) ✓✓

2. **Video retinaface exception**: Slower after 25fps conversion (+3.9s)
   - Suggests retinaface struggles with converted patient videos
   - May be over-processing standardized frames
   - Consider using mediapipe for patient video tasks

3. **Mediapipe video significantly faster**: 28% speedup is substantial
   - Makes video processing more practical (~19 videos/minute vs 14)
   - Combined with WER improvement, strongly favors 25fps for mediapipe

4. **Audio processing slightly improved**: Even audio-only benefits from standardized container

## Comparison with Standard Datasets (25fps)

| Dataset | Audio | AV mediapipe | Video mediapipe |
|---------|-------|--------------|-----------------|
| video_50_25p | 0.686s | 2.371s | 2.701s |
| video_300_25p | 0.703s | 1.743s | 2.401s |
| **patient_25p** | **0.756s** | **3.175s** | **3.162s** |

**Patient data processing**: 10-50% slower than standard datasets
- Longer videos or more complex detection scenarios
- Variable video quality requiring more processing

## Throughput Estimates (Patient 25fps Data)

### Mediapipe
- Video: ~19 videos/minute
- Audiovisual: ~19 videos/minute

### Retinaface  
- Video: ~4 videos/minute
- Audiovisual: ~4 videos/minute

### Audio-only
- ~79 audio files/minute

## Recommendations for Patient Data

1. **For speed + improved accuracy**: Use mediapipe with 25fps conversion
   - Fastest visual processing (~3s)
   - Significant WER improvement for video-only
   
2. **For best AV accuracy**: Use retinaface + 25fps (88.50% WER)
   - Slower (~14s) but best AV performance on patient data
   - 7% faster than original FPS

3. **For production/real-time**: Use audio-only (fastest, most accurate)

4. **Avoid**: Video retinaface + 25fps (slower and worse WER than original)

5. **Overall**: **25fps conversion benefits patient data for most configurations**, especially mediapipe-based workflows
