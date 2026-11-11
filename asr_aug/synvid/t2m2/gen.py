"""V2 text-to-mouth generator for semg3 (amendments A13 + A16).

text -> phonemes (CMUdict / g2p_en) -> per-phone median durations -> per-frame phone sequence
-> transformer -> latent of a frozen conv autoencoder (96x96 grey devcrop) conditioned on one reference frame.
Losses: pixel L1 + latent L1 + VGG16 perceptual + temporal smoothness + frozen Auto-AVSR VSR CTC (reader != judge).

Subcommands (all outputs under /scratch/pa2753/synvid/t2m2/gen):
  prep      build training pack from pool - T - S - heldout30 (- clips without phone timings)
  ae        train the autoencoder (then frozen)
  gen       train the generator
  sample    generate clips for a phonemized sentence list (see phonemize.py) -> mp4 + npz + rows.jsonl
"""
import argparse, json, math, os, random, subprocess, sys, time
from collections import defaultdict
from statistics import median

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SV = "/scratch/pa2753/synvid"
G = os.environ.get("T2M2_G", f"{SV}/t2m2/gen")
FFMPEG = f"{SV}/envs/latentsync/bin/ffmpeg"
AAV = f"{SV}/models/autoavsr"
PHONES = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH',
          'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
PID = {p: i + 1 for i, p in enumerate(PHONES)}  # 0 = silence
NP = len(PHONES) + 1


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ---------------------------------------------------------------- data
def frame_phones(words, T):
    ph = np.zeros(T, np.int64)
    fr = np.zeros(T, np.float32)
    for w in words:
        for p, s, e in w[3]:
            for f in range(max(0, int(math.floor(s))), min(T, int(math.ceil(e)))):
                if s <= f + 0.5 < e:
                    ph[f] = PID[p]
                    fr[f] = (f + 0.5 - s) / max(e - s, 1e-3)
    return ph, fr


def cmd_prep(a):
    os.makedirs(G, exist_ok=True)
    pool = jl(f"{SV}/t2m2/pool/pool.jsonl")
    tim = json.load(open(f"{SV}/t2m2/align/timings.json"))
    excl = {r["id"] for role in ("v0_T", "v0_S") for r in jl(f"{SV}/v0/split/roles/{role}.jsonl")}
    keep, drop = [], defaultdict(int)
    for r in pool:
        if r.get("role") != "gen_train":
            drop["heldout30" if r.get("role") == "heldout" else "pool_excluded"] += 1; continue
        if r["id"] in excl:
            drop["v0_T_or_S"] += 1; continue
        t = tim.get(r["id"])
        if t is None or any(w[3] is None for w in t["words"]):
            drop["no_phone_timing"] += 1; continue
        keep.append(r)
    rng = random.Random(20261116)
    rng.shuffle(keep)
    val_ids = {r["id"] for r in keep[:30]}
    clips, durs = [], defaultdict(list)
    lead, tail = defaultdict(list), defaultdict(list)
    for r in keep:
        t = tim[r["id"]]
        with np.load(f"{SV}/t2m2/crops/npz/{r['id']}.npz") as z:
            fr = np.clip(np.rint(z["gray96"].astype(np.float32)), 0, 255).astype(np.uint8)
        T = fr.shape[0]
        ph, frac = frame_phones(t["words"], T)
        first = int(math.floor(t["words"][0][3][0][1]))
        clips.append(dict(id=r["id"], src=r["src"], frames=torch.from_numpy(fr), ph=torch.from_numpy(ph),
                          frac=torch.from_numpy(frac), text=" ".join(w[0] for w in t["words"]),
                          lead=first, val=r["id"] in val_ids))
        if r["id"] not in val_ids:
            for w in t["words"]:
                for p, s, e in w[3]:
                    durs[p].append(e - s)
            lead[r["src"]].append(t["words"][0][3][0][1])
            tail[r["src"]].append(T - t["words"][-1][3][-1][2])
    stats = dict(phone_med={p: median(v) for p, v in durs.items()},
                 lead_med={s: median(v) for s, v in lead.items()}, tail_med={s: median(v) for s, v in tail.items()},
                 src_count={s: sum(1 for c in clips if c["src"] == s and not c["val"]) for s in lead})
    torch.save(clips, f"{G}/pack.pt")
    json.dump(stats, open(f"{G}/durations.json", "w"), indent=1)
    rep = dict(kept=len(keep), train=len(keep) - len(val_ids), val=len(val_ids), dropped=dict(drop),
               frames=int(sum(c["frames"].shape[0] for c in clips)), **{k: v for k, v in stats.items() if k != "phone_med"})
    json.dump(rep, open(f"{G}/prep_report.json", "w"), indent=1)
    print(json.dumps(rep))


# ---------------------------------------------------------------- models
def blk(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 4 if s == 2 else 3, s, 1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())


class AE(nn.Module):  # 96x96 grey <-> 8x12x12 latent
    def __init__(s):
        super().__init__()
        s.enc = nn.Sequential(blk(1, 64, 2), blk(64, 128, 2), blk(128, 256, 2), nn.Conv2d(256, 8, 3, 1, 1))
        up = lambda i, o: nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), blk(i, o, 1))
        s.dec = nn.Sequential(nn.Conv2d(8, 256, 3, 1, 1), blk(256, 256, 1), up(256, 128), up(128, 64), up(64, 32),
                              nn.Conv2d(32, 1, 3, 1, 1))

    def encode(s, x):  # x: N,1,96,96 in [0,1]
        return s.enc(x * 2 - 1)

    def decode(s, z):
        return torch.sigmoid(s.dec(z))


class Gen(nn.Module):
    def __init__(s, d=384, L=8):
        super().__init__()
        s.ph = nn.Embedding(NP, d)
        s.fr = nn.Linear(4, d)
        s.ref = nn.Linear(8 * 12 * 12, d)
        s.conv = nn.Conv1d(d, d, 5, padding=2)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 6, 4 * d, 0.1, batch_first=True, norm_first=True), L)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 8 * 12 * 12))
        s.d = d

    def pos(s, T, dev):
        p = torch.arange(T, device=dev)[:, None]
        i = torch.arange(0, s.d, 2, device=dev)[None]
        a = p / (10000 ** (i / s.d))
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(s, ph, frac, zref, mask):  # ph,frac: B,T ; zref: B,8,12,12 ; mask: B,T True=valid
        B, T = ph.shape
        f = torch.stack([torch.sin(math.pi * frac), torch.cos(math.pi * frac), frac, (ph > 0).float()], -1)
        h = s.ph(ph) + s.fr(f) + s.ref(zref.flatten(1))[:, None] + s.pos(T, ph.device)[None]
        h = h + s.conv(h.transpose(1, 2)).transpose(1, 2)
        h = s.tr(h, src_key_padding_mask=~mask)
        return zref[:, None] + s.out(h).view(B, T, 8, 12, 12)


class Reader(nn.Module):  # frozen Auto-AVSR VSR, CTC branch only
    def __init__(s, dev):
        super().__init__()
        sys.path.insert(0, f"{AAV}/code")
        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
        from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
        import sentencepiece as spm
        units = [l.split()[0] for l in open(f"{AAV}/code/spm/unigram/unigram5000_units.txt")]
        s.tok = ["<blank>"] + units + ["<eos>"]
        s.idx = {u: i for i, u in enumerate(s.tok)}
        s.sp = spm.SentencePieceProcessor(model_file=f"{AAV}/code/spm/unigram/unigram5000.model")
        s.m = E2E(len(s.tok), "video")
        s.m.load_state_dict(torch.load(f"{AAV}/vsr_trlrs2lrs3vox2avsp_base.pth", map_location="cpu"))
        s.m.eval().to(dev)
        for p in s.m.parameters():
            p.requires_grad_(False)
        s.mk = make_non_pad_mask

    def ids(s, text):
        return [s.idx[p] for p in s.sp.encode(text.upper(), out_type=str)]

    def ctc(s, x, lens, texts):  # x: B,T,96,96 in [0,1]
        x = x[:, :, 4:92, 4:92]
        x = ((x - 0.421) / 0.165)[:, :, None]  # B,T,1,88,88
        h = s.m.proj_encoder(s.m.frontend(x))
        h, _ = s.m.encoder(h, s.mk(lens).to(x.device).unsqueeze(-2))
        lp = s.m.ctc.ctc_lo(h).float().log_softmax(-1).transpose(0, 1)
        ys = [torch.tensor(s.ids(t)) for t in texts]
        yl = torch.tensor([len(y) for y in ys])
        loss = F.ctc_loss(lp, torch.cat(ys).to(x.device), lens, yl, blank=0, reduction="sum", zero_infinity=True)
        return loss / yl.sum().to(x.device), lp


class VGG(nn.Module):
    def __init__(s):
        super().__init__()
        import torchvision
        v = torchvision.models.vgg16()
        v.load_state_dict(torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")))
        s.f = v.features[:9].eval()  # through relu2_2
        for p in s.f.parameters():
            p.requires_grad_(False)
        s.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(s, a, b):  # N,1,H,W in [0,1]
        f = lambda x: s.f((x.expand(-1, 3, -1, -1) - s.mu) / s.sd)
        return F.l1_loss(f(a), f(b))


# ---------------------------------------------------------------- training
def log(path, d):
    print(json.dumps(d), flush=True)
    open(path, "a").write(json.dumps(d) + "\n")


def cmd_ae(a):
    dev = "cuda"
    torch.manual_seed(a.seed)
    clips = torch.load(f"{G}/pack.pt")
    tr = torch.cat([c["frames"] for c in clips if not c["val"]])
    va = torch.cat([c["frames"] for c in clips if c["val"]])[::7][:512]
    m = AE().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.01)
    g = torch.Generator().manual_seed(a.seed)
    lp = f"{G}/ae_log.jsonl"
    for step in range(1, a.steps + 1):
        x = tr[torch.randint(len(tr), (a.batch,), generator=g)].to(dev).float()[:, None] / 255
        with torch.autocast("cuda", torch.bfloat16):
            y = m.decode(m.encode(x)).float()
        loss = F.l1_loss(y, x) + F.l1_loss(y[..., 1:, :] - y[..., :-1, :], x[..., 1:, :] - x[..., :-1, :]) \
            + F.l1_loss(y[..., 1:] - y[..., :-1], x[..., 1:] - x[..., :-1])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == a.steps:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                xv = va.to(dev).float()[:, None] / 255
                lv = F.l1_loss(m.decode(m.encode(xv)).float(), xv).item() * 255
            log(lp, dict(step=step, loss=loss.item(), val_l1_255=lv))
    torch.save(m.state_dict(), f"{G}/ae.pth")


def batchify(cs, dev, rng, by_src):
    T = max(c["frames"].shape[0] for c in cs)
    B = len(cs)
    x = torch.zeros(B, T, 96, 96)
    ph = torch.zeros(B, T, dtype=torch.long)
    fr = torch.zeros(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    ref = torch.zeros(B, 96, 96)
    for i, c in enumerate(cs):
        n = c["frames"].shape[0]
        x[i, :n] = c["frames"].float() / 255
        ph[i, :n], fr[i, :n], mask[i, :n] = c["ph"], c["frac"], True
        o = rng.choice([d for d in by_src[c["src"]] if d is not c] or [c])
        ref[i] = o["frames"][rng.randrange(max(1, min(o["lead"], 12)))].float() / 255
    return x.to(dev), ph.to(dev), fr.to(dev), mask.to(dev), ref.to(dev)[:, None]


def cmd_gen(a):
    dev = "cuda"
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    clips = torch.load(f"{G}/pack.pt")
    tr = [c for c in clips if not c["val"]]
    va = [c for c in clips if c["val"]]
    by_src = defaultdict(list)
    for c in tr:
        by_src[c["src"]].append(c)
    ae = AE().to(dev); ae.load_state_dict(torch.load(f"{G}/ae.pth")); ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    rd, vgg = Reader(dev), VGG().to(dev)
    m = Gen().to(dev)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    ck = f"{G}/gen_resume.pth"
    step = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cpu")
        m.load_state_dict(st["m"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        rng.setstate(st["rng"])
    lp = f"{G}/gen_log.jsonl"
    t0 = time.time()
    while step < a.steps:
        step += 1
        cs = rng.sample(tr, a.batch)
        x, ph, fr, mask, ref = batchify(cs, dev, rng, by_src)
        B, T = ph.shape
        with torch.autocast("cuda", torch.bfloat16):
            zref = ae.encode(ref)
            zr = ae.encode(x.view(B * T, 1, 96, 96)).view(B, T, 8, 12, 12)
            z = m(ph, fr, zref, mask)
            y = ae.decode(z.reshape(B * T, 8, 12, 12)).view(B, T, 96, 96).float()
        mf = mask.float()
        l_pix = ((y - x).abs().mean((2, 3)) * mf).sum() / mf.sum()
        l_lat = ((z.float() - zr.float()).abs().mean((2, 3, 4)) * mf).sum() / mf.sum()
        sel = mask.view(-1).nonzero().squeeze(1)
        sel = sel[torch.randperm(len(sel), device=dev)[:256]]
        with torch.autocast("cuda", torch.bfloat16):
            l_vgg = vgg(y.view(B * T, 1, 96, 96)[sel], x.view(B * T, 1, 96, 96)[sel]).float()
        m2 = (mask[:, 1:] & mask[:, :-1]).float()
        l_sm = (((y[:, 1:] - y[:, :-1]) - (x[:, 1:] - x[:, :-1])).abs().mean((2, 3)) * m2).sum() / m2.sum()
        lam = a.lam * min(1.0, max(0.0, (step - a.ctc_start) / a.ctc_ramp))
        loss = l_pix + l_lat + 0.1 * l_vgg + l_sm
        l_ctc = torch.zeros((), device=dev)
        if lam > 0:
            with torch.autocast("cuda", torch.bfloat16):
                l_ctc, _ = rd.ctc(y, mask.sum(1), [c["text"] for c in cs])
            loss = loss + lam * l_ctc
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            log(lp, dict(step=step, loss=loss.item(), pix255=l_pix.item() * 255, lat=l_lat.item(), vgg=l_vgg.item(),
                         sm255=l_sm.item() * 255, ctc=l_ctc.item(), lam=lam, lr=sched.get_last_lr()[0], s=int(time.time() - t0)))
        if step % 2000 == 0 or step == a.steps:
            log(lp, dict(step=step, **evaluate(m, ae, rd, va, dev)))
            torch.save(dict(m=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                            rng=rng.getstate()), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    torch.save(m.state_dict(), f"{G}/gen.pth")


@torch.no_grad()
def evaluate(m, ae, rd, va, dev):
    """Gen-val monitoring only (30 pool clips; never a gate input): reconstruction L1 with true timings and
    loss-reader greedy CTC WER on the generated clip."""
    m.eval()
    rng = random.Random(0)
    by = defaultdict(list)
    for c in va:
        by[c["src"]].append(c)
    pix, E, N = [], 0, 0
    for i in range(0, len(va), 6):
        cs = va[i:i + 6]
        x, ph, fr, mask, ref = batchify(cs, dev, rng, by)
        B, T = ph.shape
        with torch.autocast("cuda", torch.bfloat16):
            y = ae.decode(m(ph, fr, ae.encode(ref), mask).reshape(B * T, 8, 12, 12)).view(B, T, 96, 96).float()
            _, lp = rd.ctc(y, mask.sum(1), [c["text"] for c in cs])
        mf = mask.float()
        pix.append((((y - x).abs().mean((2, 3)) * mf).sum() / mf.sum()).item() * 255)
        for b, c in enumerate(cs):
            seq = lp[: int(mask[b].sum()), b].argmax(-1).tolist()
            out, prev = [], 0
            for t in seq:
                if t != prev and t != 0:
                    out.append(rd.tok[t])
                prev = t
            hyp = "".join(out).replace("▁", " ").lower().split()
            ref_w = c["text"].split()
            E += lev(ref_w, hyp); N += len(ref_w)
    m.train()
    return dict(val_pix255=sum(pix) / len(pix), val_reader_wer=E / N)


def lev(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[-1]


# ---------------------------------------------------------------- sampling
def write_mp4(fr, path):
    bgr = np.repeat(fr[..., None], 3, -1)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "96x96", "-r", "25",
                    "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path + ".tmp.mp4"],
                   input=bgr.tobytes(), check=True)
    os.replace(path + ".tmp.mp4", path)


@torch.no_grad()
def cmd_sample(a):
    dev = "cuda"
    st = json.load(open(f"{G}/durations.json"))
    clips = [c for c in torch.load(f"{G}/pack.pt") if not c["val"]]
    srcs = sorted(st["src_count"])
    ae = AE().to(dev); ae.load_state_dict(torch.load(f"{G}/ae.pth")); ae.eval()
    m = Gen().to(dev); m.load_state_dict(torch.load(f"{G}/gen.pth")); m.eval()
    items = jl(a.items)  # {sid, text, phones: [[word, [ph...]], ...]}
    os.makedirs(f"{a.out}/clips", exist_ok=True); os.makedirs(f"{a.out}/npz", exist_ok=True)
    rows = []
    for k, it in enumerate(items):
        rng = random.Random(a.seed * 100003 + k)
        src = rng.choices(srcs, weights=[st["src_count"][s] for s in srcs])[0]
        rate = rng.uniform(0.9, 1.1)
        seq = [0] * max(1, round(st["lead_med"][src] * rate))
        frac = [0.0] * len(seq)
        for _, phs in it["phones"]:
            for p in phs:
                n = max(1, round(st["phone_med"][p] * rate))
                seq += [PID[p]] * n
                frac += [(j + 0.5) / n for j in range(n)]
        nt = max(1, round(st["tail_med"][src] * rate))
        seq += [0] * nt; frac += [0.0] * nt
        o = rng.choice([c for c in clips if c["src"] == src])
        ref = o["frames"][rng.randrange(max(1, min(o["lead"], 12)))].float().to(dev)[None, None] / 255
        ph = torch.tensor([seq], device=dev); fr = torch.tensor([frac], device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            z = m(ph, fr, ae.encode(ref), torch.ones_like(ph, dtype=torch.bool))
            y = ae.decode(z[0]).float()[:, 0]
        u8 = (y.clamp(0, 1) * 255).round().byte().cpu().numpy()
        p = f"{a.out}/clips/{it['sid']}.mp4"
        write_mp4(u8, p)
        np.savez_compressed(f"{a.out}/npz/{it['sid']}.npz", gray96=u8.astype(np.float16), missing=np.zeros(len(u8), bool))
        rows.append(dict(sid=it["sid"], take_id=it["sid"], video_path=p, npz_path=f"{a.out}/npz/{it['sid']}.npz",
                         transcript=it["text"], words=len(it["text"].split()), frames=len(u8), src_ref=o["id"],
                         valid=bool(u8.shape[1:] == (96, 96) and np.isfinite(y.cpu().numpy()).all())))
    with open(f"{a.out}/rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(dict(n=len(rows), valid=sum(r["valid"] for r in rows))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "ae", "gen", "sample"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--ctc-start", type=int, default=2000)
    ap.add_argument("--ctc-ramp", type=int, default=4000)
    ap.add_argument("--items")
    ap.add_argument("--out")
    a = ap.parse_args()
    dict(prep=cmd_prep, ae=cmd_ae, gen=cmd_gen, sample=cmd_sample)[a.cmd](a)


if __name__ == "__main__":
    main()
