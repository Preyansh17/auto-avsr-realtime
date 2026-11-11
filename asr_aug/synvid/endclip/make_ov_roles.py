"""s3ov_val role files pointing at the readable devcrop front-end variants (devcrop_v1 itself is not readable for pa2753).
The variant used for the do-no-harm check is the one whose arm-a decode reproduces D0's s3ov_val predictions."""
import json, os
B = "/scratch/th3482/Dysarthric_ASR/data/patients/derived"
rows = [json.loads(l) for l in open("/scratch/th3482/Dysarthric_ASR/data/patients/splits/s3deliver_v8/roles/dc_s3ov_val.jsonl")]
for v in ("fi_v1", "fu_v1", "fv_v1"):
    d = f"/scratch/pa2753/synvid/endclip/split_ov_{v}/roles"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/dc_s3ov_val.jsonl", "w") as f:
        for r in rows:
            p = r["video_path"].replace("/devcrop_v1/", f"/devcrop_{v}/"); assert os.path.exists(p), p
            f.write(json.dumps(dict(r, video_path=p)) + "\n")
print("ok", len(rows))
