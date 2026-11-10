"""A13 step 0.2-0.5 analysis (CPU): crop success + spot check, word/phone timings, vocabulary coverage, text pool.

Inputs: t2m2/pool/pool.jsonl, t2m2/crops/npz, t2m2/align/video_align.jsonl, Tianyu's in-home predictions (read-only),
DailyDialog from the text package (read-only). Output: t2m2/step0_stats.json, t2m2/text/*.
"""
import argparse, collections, csv, gzip, json, os, random, re, sys

import numpy as np

SV = "/scratch/pa2753/synvid"
T2 = f"{SV}/t2m2"
sys.path.insert(0, f"{SV}/code"); sys.path.insert(0, f"{SV}/pbias"); sys.path.insert(0, f"{T2}/code")
import pbias  # noqa: E402
from pool import EVAL, EVAL_LC, LC, R, normalize  # noqa: E402

# copied from t2m_build.py (A11; importing it pulls cv2, which needs libGL, absent on cs nodes)
AVG_DUR = dict(AA=.12, AE=.12, AH=.07, AO=.12, AW=.15, AY=.14, EH=.09, ER=.11, EY=.12, IH=.07, IY=.10, OW=.12, OY=.16,
               UH=.07, UW=.10, B=.06, CH=.09, D=.05, DH=.04, F=.09, G=.06, HH=.06, JH=.08, K=.07, L=.06, M=.07, N=.06,
               NG=.07, P=.07, R=.06, S=.10, SH=.11, T=.06, TH=.09, V=.06, W=.06, Y=.06, Z=.08, ZH=.08)


def load_cmu(path):
    d = {}
    for l in open(path, encoding="latin-1"):
        p = l.split()
        if len(p) >= 3 and p[0].lower() not in d:
            d[p[0].lower()] = [re.sub(r"\d", "", x) for x in p[2:]]
    return d


TXT = "/scratch/th3482/Dysarthric_ASR/data/patients/texts/dialog_corpora_v20261031/corpus"
V34_TY = "/scratch/th3482/Dysarthric_ASR/data/patients/derived/devcrop_v34_v1"
FPS = 25.0


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def q(v, ps=(10, 50, 90)):
    v = np.asarray(v, float)
    return {f"p{p}": round(float(np.percentile(v, p)), 3) for p in ps} | dict(mean=round(float(v.mean()), 3), n=int(len(v)))


def active_span(ap):
    """Tianyu's devcrop_v1 onset rule (INDEX.md.devcrop): median-3, baseline p10, r = p90 - p10,
    active if > b + 0.25 r or |vel| > 0.08 r, runs >= 3 frames, gaps <= 4 merged. Returns (first, last_excl) or None."""
    a = np.asarray(ap, float)
    ok = np.isfinite(a)
    if ok.sum() < 5: return None
    a = np.interp(np.arange(len(a)), np.flatnonzero(ok), a[ok])
    m = np.array([np.median(a[max(0, i - 1):i + 2]) for i in range(len(a))])
    b, t = np.percentile(m, 10), np.percentile(m, 90); r = max(t - b, 1e-6)
    act = (m > b + 0.25 * r) | (np.abs(np.gradient(m)) > 0.08 * r)
    runs, i = [], 0
    while i < len(act):
        if act[i]:
            j = i
            while j < len(act) and act[j]: j += 1
            if j - i >= 3: runs.append([i, j])
            i = j
        else:
            i += 1
    if not runs: return None
    mg = [runs[0]]
    for s, e in runs[1:]:
        if s - mg[-1][1] <= 4: mg[-1][1] = e
        else: mg.append([s, e])
    return mg[0][0], mg[-1][1]


def crops_section(pool, out):
    keep = [p for p in pool if "drop" not in p]
    log = {}
    if os.path.exists(f"{T2}/crops/devcrop_log.jsonl"):
        for x in jl(f"{T2}/crops/devcrop_log.jsonl"): log[x["id"]] = x
    res = dict(by_src={}, failed=[])
    T, miss, srcfps = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for p in keep:
        f = f"{T2}/crops/npz/{p['id']}.npz"
        if not os.path.exists(f):
            res["failed"].append(p["id"]); continue
        z = np.load(f); n = len(z["missing"]); m = float(z["missing"].mean()) if n else 1.0
        if n == 0 or m > 0.5: res["failed"].append(p["id"])
        T[p["src"]].append(n); miss[p["src"]].append(m)
        if p["id"] in log and n: srcfps[p["src"]].append(log[p["id"]]["source_frames"] / (n / FPS))
    for s in sorted({p["src"] for p in keep}):
        nk = sum(p["src"] == s for p in keep)
        ok = sum(1 for p in keep if p["src"] == s and p["id"] not in res["failed"] and os.path.exists(f"{T2}/crops/npz/{p['id']}.npz"))
        res["by_src"][s] = dict(clips=nk, ok=ok, success=round(ok / nk, 4), frames=q(T[s]), sec=q(np.array(T[s]) / FPS),
                                missing_frac=q(miss[s], (50, 90, 99)), clips_any_missing=int(sum(x > 0 for x in miss[s])),
                                total_frames=int(sum(T[s])), source_fps=q(srcfps[s]) if srcfps[s] else None)
    res["total_ok"] = sum(v["ok"] for v in res["by_src"].values()); res["total"] = len(keep)
    res["total_frames"] = sum(v["total_frames"] for v in res["by_src"].values())
    # spot check: 10 v34 clips vs Tianyu's devcrop_v34_v1 (the only readable pre-existing devcrops of pool clips)
    rng = random.Random(20261113); v = sorted(p["id"] for p in keep if p["src"] == "v34")
    sc = []
    for i in rng.sample(v, 10):
        mine = np.load(f"{T2}/crops/npz/{i}.npz"); tid = i.split("__", 1)[1]
        ty_f = f"{V34_TY}/v34ph_{tid[len('SEMG3_'):]}.npz"
        if not os.path.exists(ty_f): sc.append(dict(id=i, error="no tianyu crop")); continue
        ty = np.load(ty_f); a, b = mine["gray96"].astype(np.float32), ty["gray96"].astype(np.float32)
        best = None
        for off in range(-40, 41):  # b[k] ~ a[k + off]
            lo, hi = max(0, -off), min(len(b), len(a) - off)
            if hi - lo < 10: continue
            mae = float(np.abs(a[lo + off:hi + off] - b[lo:hi]).mean())
            if best is None or mae < best[1]: best = (off, mae, hi - lo)
        sc.append(dict(id=i, T_mine=len(a), T_tianyu=len(b), best_offset=best[0], mae_gray=round(best[1], 3),
                       overlap=best[2], exact_frames=int(sum(np.array_equal(a[k + best[0]], b[k]) for k in range(max(0, -best[0]), min(len(b), len(a) - best[0]))))))
    res["spot_check_v34"] = sc
    out["crops"] = res


def timing_section(pool, cmu, out):
    keep = {p["id"]: p for p in pool if "drop" not in p}
    al = {x["id"]: x for x in jl(f"{T2}/align/video_align.jsonl")} if os.path.exists(f"{T2}/align/video_align.jsonl") else {}
    res = dict(aligned=sum("words" in x for x in al.values()), errors=[x["id"] for x in al.values() if "error" in x])
    spw, lead, tail, lead_ap, tail_ap, wdur = [collections.defaultdict(list) for _ in range(6)]
    phone = collections.defaultdict(list); oov = collections.Counter(); timings = {}
    for i, x in al.items():
        if "words" not in x or i not in keep: continue
        src = keep[i]["src"]; W = x["words"]; Tn = x["frames"]
        b = [W[0][1]] + [(W[k][2] + W[k + 1][1]) / 2 for k in range(len(W) - 1)] + [W[-1][2]]
        spw[src].append((b[-1] - b[0]) / FPS / len(W)); lead[src].append(b[0] / FPS); tail[src].append((Tn - b[-1]) / FPS)
        z = np.load(f"{T2}/crops/npz/{i}.npz"); sp = active_span(z["aperture"])
        if sp: lead_ap[src].append(sp[0] / FPS); tail_ap[src].append((len(z["aperture"]) - sp[1]) / FPS)
        wt = []
        for k, (w, _, _) in enumerate(W):
            d = (b[k + 1] - b[k]) / FPS; wdur[src].append(d)
            ph = cmu.get(w)
            if ph is None:
                oov[w] += 1; wt.append([w, round(b[k], 2), round(b[k + 1], 2), None]); continue
            r = np.array([AVG_DUR.get(p, .08) for p in ph]); r = r / r.sum()
            pd = [[p, round(float(b[k] + (b[k + 1] - b[k]) * c0), 2), round(float(b[k] + (b[k + 1] - b[k]) * c1), 2)]
                  for p, c0, c1 in zip(ph, np.concatenate([[0], np.cumsum(r)[:-1]]), np.cumsum(r))]
            for p, s, e in pd: phone[p].append((e - s) / FPS)
            wt.append([w, round(b[k], 2), round(b[k + 1], 2), pd])
        timings[i] = dict(frames=Tn, words=wt, active_span=sp)
    json.dump(timings, open(f"{T2}/align/timings.json", "w"))
    allv = lambda d: [v for s in d for v in d[s]]  # noqa: E731
    res["sec_per_word"] = {s: q(v) for s, v in spw.items()} | dict(all=q(allv(spw)))
    res["word_dur_sec"] = q(allv(wdur))
    res["leadin_ctc_sec"] = {s: q(v) for s, v in lead.items()} | dict(all=q(allv(lead)))
    res["tail_ctc_sec"] = {s: q(v) for s, v in tail.items()} | dict(all=q(allv(tail)))
    res["leadin_aperture_sec"] = {s: q(v) for s, v in lead_ap.items() if v} | dict(all=q(allv(lead_ap)))
    res["tail_aperture_sec"] = {s: q(v) for s, v in tail_ap.items() if v} | dict(all=q(allv(tail_ap)))
    res["phone_dur_sec"] = {p: dict(q(v, (10, 50, 90)), prior=AVG_DUR.get(p)) for p, v in sorted(phone.items())}
    res["cmudict_oov_words"] = dict(types=len(oov), tokens=sum(oov.values()), words=sorted(oov))
    out["timings"] = res


def vocab_section(pool, out):
    keep = [p for p in pool if "drop" not in p]
    V = {s: set() for s in ("s3", "s3ov", "v34", "real")}
    Vg = set()
    for p in keep:
        V[p["src"]] |= set(pbias.W(p["norm"]))
        if p["role"] == "gen_train": Vg |= set(pbias.W(p["norm"]))
    Vall = set().union(*V.values())
    # F1 training text = all s3_train + s3ov_train + real117 rows (deliverable); v34 is new to F1
    F1v = set()
    for role in ("dc_s3_train", "dc_s3ov_train", "dc_real_all"):
        for x in jl(f"{R}/{role}.jsonl"): F1v |= set(pbias.W(normalize(x["transcript"])))
    S, SR = pbias.S, pbias.SR
    G = {x["rel_path"]: pbias.W(x["normalized_ground_truth"])
         for x in json.load(open(S / "bp/decgrid_F1/c0.1_b10/evaluation/dc_home_all.predictions.json"))}
    val, trainw = {}, {}
    for k in pbias.K:
        tr = [G[x["rel_path"]] for x in pbias.jl(SR / f"rcv{k}_home_train.jsonl")]
        for x in pbias.jl(SR / f"rcv{k}_home_val.jsonl"): val[x["rel_path"]] = k
        trainw[k] = {w for t in tr for w in t}
    chain = pbias.pools_chain(S / "bp/chain/F1_app45_q8/fus_lm0.1_lb0/evaluation/dc_home_all.predictions.json",
                              S / "lm/F1_chainq8_v3/Qwen2.5-0.5B.json")
    unseen, unseen_err, E = [], [], 0
    for rp, k in val.items():
        r = G[rp]; e, bad = pbias.align(r, pbias.pick(chain[rp][1], 0, set(), set())); E += e
        for w, bd in zip(r, bad):
            if w not in trainw[k]:
                unseen.append(w)
                if bd: unseen_err.append(w)
    assert E == 81 and len(unseen) == 231, (E, len(unseen))

    def cov(tokens):
        c = dict(tokens=len(tokens), types=len(set(tokens)))
        for name, vs in (("pool_all", Vall), ("pool_gen_train", Vg), ("F1_train_text", F1v), ("v34_train", V["v34"]),
                         ("conversation_real", V["real"])):
            c[name] = sum(w in vs for w in tokens)
        c["v34_only_new_to_F1"] = sum(w in V["v34"] and w not in F1v for w in tokens)
        c["not_in_pool"] = sorted(collections.Counter(w for w in tokens if w not in Vall).items(), key=lambda x: -x[1])
        return c
    allt = [w for rp in val for w in G[rp]]
    out["vocab"] = dict(pool_types=len(Vall), pool_gen_train_types=len(Vg), by_src={s: len(v) for s, v in V.items()},
                        F1_train_text_types=len(F1v), chain_errors=E,
                        inhome_all_1116=cov(allt), unseen_231=cov(unseen), chain_unseen_errors=cov(unseen_err))
    return Vg, G, unseen


def sentences(text):
    for s in re.split(r"(?<=[.?!])\s+|(?<=[.?!])(?=[A-Z])", text.replace("’", "'")):
        s = s.strip()
        if s: yield s


def textpool_section(pool, Vg, G, unseen, cmu, out):
    os.makedirs(f"{T2}/text", exist_ok=True)
    ev_exact, ev_norm = set(), set()
    for role in EVAL:
        for x in jl(f"{R}/{role}.jsonl"):
            ev_exact.add(x["transcript"].strip().lower()); ev_norm.add(normalize(x["transcript"]))
            if x.get("raw_sentence"): ev_exact.add(x["raw_sentence"].strip().lower()); ev_norm.add(normalize(x["raw_sentence"]))
    for role in EVAL_LC:
        for x in jl(f"{LC}/{role}.jsonl"):
            for k in ("transcript", "raw_sentence"):
                if x.get(k): ev_exact.add(x[k].strip().lower()); ev_norm.add(normalize(x[k]))
    for t in G.values(): ev_norm.add(" ".join(t))
    n_pkg = 0
    for x in csv.DictReader(open(f"{TXT}/test_val_texts.tsv"), delimiter="\t"):  # every v3.x / oct16 val+test sentence
        ev_exact.add(x["text"].strip().lower()); ev_norm.add(normalize(x["text"])); n_pkg += 1
        for k in x["keys"].split(" | "): ev_norm.add(normalize(k))
    held = {l.split("\t")[1].strip() for l in open(f"{T2}/pool/heldout30.txt")}
    ev_norm_ns = {n.replace(" ", "") for n in ev_norm}
    stats = collections.Counter(); cand = {}
    for x in (json.loads(l) for l in gzip.open(f"{TXT}/dailydialog.jsonl.gz", "rt")):
        if x.get("test_val_dialog_in"):
            stats["dialogs_dropped_test_val"] += 1; continue
        stats["dialogs_used"] += 1
        for t in x["turns"]:
            if t.get("same_text_test_val"): stats["turns_flagged"] += 1; continue
            for s in sentences(t["text"]):
                n = normalize(s); nw = len(n.split())
                if not 3 <= nw <= 10: continue
                stats["sent_3to10"] += 1
                if s.strip().lower() in ev_exact: stats["leak_exact"] += 1; continue
                if n in ev_norm or n.replace(" ", "") in ev_norm_ns: stats["leak_norm"] += 1; continue
                if n in held: stats["heldout30"] += 1; continue
                if any(w not in cmu for w in n.split()): stats["cmu_oov"] += 1; continue
                miss = [w for w in n.split() if w not in Vg]
                if not miss: stats["all_words_in_video"] += 1; continue
                cand.setdefault(n, (s, miss))
    # re-check: no candidate may match any eval text (fails hard)
    for n, (s, _) in cand.items():
        assert n not in ev_norm and s.strip().lower() not in ev_exact, n
    tw = collections.Counter(w for _, m in cand.values() for w in set(m))
    # greedy 500: maximise new target words, ties -> more frequent words, then shorter
    chosen, cov = [], set(); items = sorted(cand.items())
    while len(chosen) < 500:
        best = max(items, key=lambda it: (len(set(it[1][1]) - cov), sum(tw[w] for w in set(it[1][1]) - cov), -len(it[0].split())))
        if not set(best[1][1]) - cov: break
        chosen.append(best); cov |= set(best[1][1])
    with open(f"{T2}/text/dd_pool.jsonl", "w") as f:
        for n, (s, m) in sorted(cand.items()): f.write(json.dumps(dict(text=s, norm=n, new_words=m)) + "\n")
    with open(f"{T2}/text/dd_greedy500.jsonl", "w") as f:
        for n, (s, m) in chosen: f.write(json.dumps(dict(text=s, norm=n, new_words=m)) + "\n")
    un_out = [w for w in unseen if w not in Vg]
    out["text_pool"] = dict(source="DailyDialog (CC BY-NC-SA 4.0) from the 2026-10-31 text package", filter=dict(stats),
                            eval_texts_checked=dict(exact=len(ev_exact), normalised=len(ev_norm), package_val_test_rows=n_pkg),
                            pool_sentences=len(cand), target_words_covered=len(tw),
                            greedy500=dict(sentences=len(chosen), target_words=len(cov)),
                            inhome_unseen_not_in_gen_video=dict(tokens=len(un_out),
                                                                covered_by_text_pool=sum(w in tw for w in un_out),
                                                                covered_by_greedy500=sum(w in cov for w in un_out)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="crops,timings,vocab,text"); a = ap.parse_args()
    pool = jl(f"{T2}/pool/pool.jsonl"); cmu = load_cmu("/home/pa2753/nltk_data/corpora/cmudict/cmudict")
    out = {}
    parts = a.parts.split(",")
    if "crops" in parts: crops_section(pool, out)
    if "timings" in parts: timing_section(pool, cmu, out)
    if "vocab" in parts or "text" in parts:
        Vg, G, unseen = vocab_section(pool, out)
        if "text" in parts: textpool_section(pool, Vg, G, unseen, cmu, out)
    f = f"{T2}/step0_stats.json"
    old = json.load(open(f)) if os.path.exists(f) else {}
    old.update(out); json.dump(old, open(f, "w"), indent=1)
    print(json.dumps(out, indent=1)[:20000])


if __name__ == "__main__":
    main()
