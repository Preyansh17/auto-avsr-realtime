"""Phase 3 pilot (plan 3.4 part 1, A9): pick 100 sentences and 100 distinct s3_train templates, then run a generator.

  select  100 clips from data/drive.json with random.Random(20261108).sample (fixed seed, no look at the words).
          Templates: s3_train rows in random.Random(20261109) order; longest driving clip first, each takes the first
          unused template whose length (frame_count / 25) is at least the driving clip; if none is long enough it takes
          the longest unused one and is flagged pingpong (the generator then loops the template).
          Out: pilot/pilot.jsonl {pid, text, words, drive_wav, drive_sec, voice, template_tid, template_path,
          template_text, template_frames, pingpong}.
  ls      LatentSync 1.6 pipeline loaded once (cwd = LatentSync checkout), one clip per pilot row, same settings as the
          released inference script (20 DDIM steps, guidance 1.5, DeepCache, seed 1247). --ckpt picks zero-shot
          (checkpoints/latentsync_unet.pt) or G-LSft. Out: <out-dir>/<pid>.mp4 + gen_log.jsonl (seconds per clip).
KeySync runs through its own scripts/infer_raw.sh (see pilot_gen.sbatch).
"""
import argparse, json, os, random, sys, time

ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"


def norm25(row):
    real = os.path.realpath(row["video_path"])
    return real.split("/processed/v1/")[0] + "/processed/v1/_normalized_25p/" + row["take_id"] + ".mp4"


def cmd_select(a):
    drive = json.load(open(a.drive))
    pick = random.Random(20261108).sample(drive, a.n)
    rows = [json.loads(l) for l in open(f"{ROLES}/s3_train.jsonl") if l.strip()]
    for r in rows:
        r.setdefault("take_id", os.path.basename(r["video_path"])[:-4])
    random.Random(20261109).shuffle(rows)
    used, out = set(), []
    for d in sorted(pick, key=lambda x: -x["sec"]):
        cand = [r for r in rows if r["take_id"] not in used and r["frame_count"] / 25.0 >= d["sec"]]
        pp = not cand
        t = cand[0] if cand else max((r for r in rows if r["take_id"] not in used), key=lambda r: r["frame_count"])
        used.add(t["take_id"])
        out.append(dict(pid=f"p{len(out):03d}", id=d["id"], text=d["text"], words=len(d["text"].split()),
                        drive_wav=d["wav"], drive_sec=d["sec"], voice=d["voice"], template_tid=t["take_id"],
                        template_path=norm25(t), template_text=t["transcript"], template_frames=t["frame_count"],
                        pingpong=pp))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"{len(out)} pilot rows, {len(used)} distinct templates, pingpong {sum(o['pingpong'] for o in out)}")


def cmd_ls(a):
    import torch
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from accelerate.utils import set_seed
    from DeepCache import DeepCacheSDHelper
    sys.path.insert(0, os.getcwd())
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    config = OmegaConf.load("configs/unet/stage2_512.yaml")
    dtype = torch.float16
    sched = DDIMScheduler.from_pretrained("configs")
    audio_encoder = Audio2Feature(model_path="checkpoints/whisper/tiny.pt", device="cuda",
                                  num_frames=config.data.num_frames, audio_feat_length=config.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor, vae.config.shift_factor = 0.18215, 0
    unet, step = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(config.model), a.ckpt, device="cpu")
    pipe = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet.to(dtype=dtype), scheduler=sched).to("cuda")
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, "gen_log.jsonl"), "a")
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    for r in rows:
        out = os.path.join(a.out_dir, f"{r['pid']}.mp4")
        if os.path.exists(out):
            continue
        set_seed(1247)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        rec = dict(pid=r["pid"], ckpt=a.ckpt, ckpt_step=step)
        try:
            pipe(video_path=r["template_path"], audio_path=r["drive_wav"], video_out_path=out + ".tmp.mp4",
                 num_frames=config.data.num_frames, num_inference_steps=20, guidance_scale=1.5, weight_dtype=dtype,
                 width=config.data.resolution, height=config.data.resolution,
                 mask_image_path=config.data.mask_image_path, temp_dir=os.path.join(a.out_dir, "tmp"))
            os.replace(out + ".tmp.mp4", out)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec.update(sec=round(time.time() - t0, 2), peak_mib=int(torch.cuda.max_memory_allocated() / 2 ** 20))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select"); s.add_argument("--drive", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=100)
    l = sp.add_parser("ls"); l.add_argument("--rows", required=True); l.add_argument("--ckpt", required=True)
    l.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    dict(select=cmd_select, ls=cmd_ls)[a.cmd](a)


if __name__ == "__main__":
    main()
