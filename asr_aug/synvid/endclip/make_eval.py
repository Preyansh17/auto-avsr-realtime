"""Build the padded-eval copy of the delivered v1.8 usr2_eval.py: the only change repeats the last input frame
PAD_FRAMES times (env var, 25 fps) after the crop is loaded. Also builds the role files for the two decode sets."""
import os, shutil
D = "/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105/code/hpc/train_snapshot_849ef5f"
O = "/scratch/pa2753/synvid/endclip/code"; os.makedirs(O, exist_ok=True)
shutil.copy(f"{D}/s3d_video.py", O)
src = open(f"{D}/usr2_eval.py").read()
site = "            T = v.shape[0]\n"
assert src.count(site) == 1
patch = ("            _pad = int(os.environ.get('PAD_FRAMES', '0'))  # endclip: repeat the last frame\n"
         "            if _pad > 0:\n"
         "                v = torch.cat([v, v[-1:].repeat(_pad, *([1] * (v.dim() - 1)))], 0)\n" + site)
open(f"{O}/usr2_eval_pad.py", "w").write(src.replace(site, patch))
R = "/scratch/pa2753/synvid/endclip/split/roles"; os.makedirs(R, exist_ok=True)
shutil.copy("/scratch/th3482/Dysarthric_ASR/data/patients/splits/semg3_rcv_20261109_v1/roles/dc_home_all.jsonl", R)
shutil.copy("/scratch/th3482/Dysarthric_ASR/data/patients/splits/s3deliver_v8/roles/dc_s3ov_val.jsonl", R)
print("ok")
