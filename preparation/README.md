# preparation/

Face and mouth-ROI detectors used by the streaming pipeline.

The upstream LRS2 / LRS3 / VoxCeleb2 corpus-preparation scripts have been removed — this project
trains on pre-cropped patient clips, not on those corpora. What remains is the detector code that
`demo_realtime.py`, `eval.py`, `online_avsr/streaming.py`, and `scripts/face_crop_patient.py`
import at runtime.

| Path | Purpose |
| --- | --- |
| `detectors/mediapipe/` | auto_avsr mouth-ROI detector (Apache 2.0) |
| `detectors/mediapipe_face/` | torchaudio data_prep face-crop detector — what the pretrained `device` model expects |
| `detectors/retinaface/` | RetinaFace mouth-ROI detector, used for the patient crops |
| `data/data_module.py` | `AVSRDataLoader` — video/audio loading shared with the root entry points |
| `transforms.py`, `utils.py` | Shared preprocessing helpers |

Selected at runtime via `--preprocess {face, mouth, roi, none}`. Patient clips are already cropped,
so use `--preprocess roi` end to end and do **not** re-run detection on them.

The mouth-ROI detectors expect `20words_mean_face.npy` (shipped alongside each detector) for
landmark alignment.
