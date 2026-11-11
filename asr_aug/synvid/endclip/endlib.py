"""End-of-clip check helpers: word alignment with per-reference-word status, first/last-word errors, paired bootstrap."""
import random, re, sys

D = "/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105"
sys.path.insert(0, f"{D}/code/hpc/third_party/dasr_metrics")
from dysarthric_asr.evaluation.metrics import normalize  # the delivered normaliser (strips "%")


def normalize_pct(t):
    """Consistent normaliser: '%' read as the word 'percent' before the delivered normaliser runs."""
    return normalize(re.sub(r"\s*%", " percent ", t))


def align(r, h):
    """Levenshtein with backtrace. Returns (subs, dels, ins, ok) where ok[i] says whether ref word i is matched."""
    n, m = len(r), len(h)
    Dm = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): Dm[i][0] = i
    for j in range(m + 1): Dm[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            Dm[i][j] = min(Dm[i - 1][j] + 1, Dm[i][j - 1] + 1, Dm[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, s, d, ins = n, m, 0, 0, 0
    ok = [False] * n
    while i or j:
        if i and j and Dm[i][j] == Dm[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ok[i - 1] = r[i - 1] == h[j - 1]; s += r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif i and Dm[i][j] == Dm[i - 1][j] + 1:
            d += 1; i -= 1
        else:
            ins += 1; j -= 1
    return s, d, ins, ok


def clip_stats(ref, hyp, norm=normalize):
    r, h = norm(ref).split(), norm(hyp).split()
    s, d, i, ok = align(r, h)
    return dict(err=s + d + i, words=len(r), first_wrong=int(bool(r) and not ok[0]), last_wrong=int(bool(r) and not ok[-1]),
                last_ref=r[-1] if r else "", hyp_last=h[-1] if h else "")


def summarize(stats):
    return dict(clips=len(stats), last_wrong=sum(x["last_wrong"] for x in stats), first_wrong=sum(x["first_wrong"] for x in stats),
                err=sum(x["err"] for x in stats), words=sum(x["words"] for x in stats))


def paired_bootstrap(a, b, keys=("last_wrong", "err"), n=10000, seed=0):
    """a, b: lists of per-clip stats in the same clip order. CI of sum(b) - sum(a) per key."""
    assert len(a) == len(b)
    rng = random.Random(seed); N = len(a); out = {}
    diffs = {k: [y[k] - x[k] for x, y in zip(a, b)] for k in keys}
    samples = {k: [] for k in keys}
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        for k in keys:
            samples[k].append(sum(diffs[k][i] for i in idx))
    for k in keys:
        s = sorted(samples[k])
        out[k] = dict(diff=sum(diffs[k]), lo=s[int(0.025 * n)], hi=s[int(0.975 * n) - 1])
    return out
