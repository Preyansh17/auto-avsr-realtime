"""Step 1.4 check: rebuilt crops vs Tianyu's crops for the same takes.

Frame counts, mean absolute pixel difference (uint8, all channels, per clip), and judge WER on both sets
(from two judge.py JSONL outputs). Pass: frame counts match on all clips and |WER_rebuilt - WER_orig| <= 2 pp
(pooled WER = total errors / total reference words).

Usage: python compare_crops.py --rows s3_train.jsonl --rebuilt-dir OUT/crops --judge-orig A.jsonl
       --judge-rebuilt B.jsonl --out cmp.json
"""
import argparse, json, os

import cv2
import numpy as np


def frames(p):
    c = cv2.VideoCapture(p)
    fr = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fr.append(f)
    c.release()
    return np.stack(fr) if fr else np.zeros((0, 96, 96, 3), np.uint8)


def pooled(path, ids):
    rs = [json.loads(l) for l in open(path) if l.strip()]
    rs = [r for r in rs if r["take_id"] in ids]
    return sum(r["errors"] for r in rs) / max(1, sum(r["ref_words"] for r in rs)), len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--rebuilt-dir", required=True)
    ap.add_argument("--judge-orig", required=True)
    ap.add_argument("--judge-rebuilt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {json.loads(l)["take_id"]: json.loads(l) for l in open(a.rows) if l.strip()}
    per = []
    for f in sorted(os.listdir(a.rebuilt_dir)):
        tid = f[:-4]
        A, B = frames(rows[tid]["video_path"]), frames(os.path.join(a.rebuilt_dir, f))
        n = min(len(A), len(B))
        mad = float(np.abs(A[:n].astype(np.int16) - B[:n].astype(np.int16)).mean()) if n else None
        per.append(dict(take_id=tid, frames_orig=len(A), frames_rebuilt=len(B), match=len(A) == len(B), mad=mad))
    ids = {p["take_id"] for p in per}
    wo, no = pooled(a.judge_orig, ids)
    wr, nr = pooled(a.judge_rebuilt, ids)
    res = dict(n=len(per), frames_match=sum(p["match"] for p in per),
               mad_mean=float(np.mean([p["mad"] for p in per if p["mad"] is not None])),
               mad_max=float(np.max([p["mad"] for p in per if p["mad"] is not None])),
               judge_wer_orig=wo, judge_wer_rebuilt=wr, judge_n=(no, nr), diff_pp=100 * (wr - wo),
               passed=bool(all(p["match"] for p in per) and abs(wr - wo) <= 0.02 and no == nr == len(per)),
               per_clip=per)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))


if __name__ == "__main__":
    main()
