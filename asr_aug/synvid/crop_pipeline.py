"""Rebuilt semg3 mouth-crop pipeline (plan A2) + FAN inner-lip landmarks (A3).

Same path as Tianyu's preprocess (AVSRDataLoader, detector=retinaface, convert_gray=False -> save_vid_aud_txt):
  torchvision.io.read_video (RGB, THWC uint8) of the 25 fps full-frame copy
  -> preparation.detectors.retinaface LandmarksDetector (ibug RetinaFace resnet50 thr 0.8 + FAN 68 points, largest face)
  -> VideoProcess(convert_gray=False): landmark interpolation, 12-frame smoothing window, similarity warp to the
     20words mean face (stable points 28,33,36,39,42,45,48,54) at 256x256, 96x96 patch around mouth points 48-67
  -> torchvision.io.write_video(fps=25) (libx264, as preparation.utils.save2vid).
Per clip it also saves the raw FAN landmarks (T,68,2, NaN where no face) as .npy.

Usage (GPU node):
  python crop_pipeline.py --rows s3_train.jsonl --take-ids ids.txt --out-dir OUT [--limit N] [--shard i --nshard n]
Inputs are rows of a role JSONL; the source video is <session>/processed/v1/_normalized_25p/<take_id>.mp4, or --src-dir.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("PREP_ROOT", HERE))  # dir containing the repo's preparation/ package


def norm25_path(row):
    # video_path = .../<session>/processed/v1/<dataset>/<..._video_seg24s>/all/<take>.mp4 (via split symlink)
    real = os.path.realpath(row["video_path"])
    v1 = real.split("/processed/v1/")[0] + "/processed/v1"
    return os.path.join(v1, "_normalized_25p", row["take_id"] + ".mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--take-ids", default=None, help="file with one take_id per line (subset)")
    ap.add_argument("--src-dir", default=None, help="read <src-dir>/<take_id>.mp4 instead of _normalized_25p")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()

    from preparation.detectors.retinaface.detector import LandmarksDetector
    from preparation.detectors.retinaface.video_process import VideoProcess

    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    if a.take_ids:
        keep = [l.strip() for l in open(a.take_ids) if l.strip()]
        byid = {r["take_id"]: r for r in rows}
        rows = [byid[t] for t in keep]
    rows = rows[a.shard::a.nshard][: a.limit]
    os.makedirs(os.path.join(a.out_dir, "crops"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "landmarks"), exist_ok=True)
    det = LandmarksDetector(device="cuda:0" if torch.cuda.is_available() else "cpu")
    vp = VideoProcess(convert_gray=False)
    log = open(os.path.join(a.out_dir, f"crop_log_{a.shard}.jsonl"), "a")
    for r in rows:
        tid = r["take_id"]
        out = os.path.join(a.out_dir, "crops", tid + ".mp4")
        if os.path.exists(out):
            continue
        src = os.path.join(a.src_dir, tid + ".mp4") if a.src_dir else norm25_path(r)
        t0 = time.time()
        video = torchvision.io.read_video(src, pts_unit="sec")[0].numpy()
        lms = det(video)
        miss = sum(l is None for l in lms)
        arr = np.full((len(lms), 68, 2), np.nan, np.float32)
        for i, l in enumerate(lms):
            if l is not None:
                arr[i] = l
        np.save(os.path.join(a.out_dir, "landmarks", tid + ".npy"), arr)
        crop = vp(video, [None if l is None else l.copy() for l in lms])
        ok = crop is not None
        if ok:
            torchvision.io.write_video(out, torch.tensor(crop), 25)
        log.write(json.dumps(dict(take_id=tid, src=src, frames_in=len(video), frames_out=len(crop) if ok else 0,
                                  no_face_frames=miss, ok=ok, sec=round(time.time() - t0, 2))) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
