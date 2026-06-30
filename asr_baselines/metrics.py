"""WER/CER for the audio ASR baselines.

Normalization is intentionally light and shared across all models (Emformer,
Whisper, Nemotron) so the numbers are comparable: lowercase, strip punctuation,
collapse whitespace. (Whisper's own EnglishTextNormalizer is heavier; keep one
normalizer for the whole comparison rather than per-model ones.)
"""

import re
import string


_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def compute_wer_cer(refs, hyps):
    """Returns dict with wer, cer, n_words, n_chars. Uses jiwer if available,
    else a builtin Levenshtein (so this imports without the HPC deps)."""
    refs = [normalize(r) for r in refs]
    hyps = [normalize(h) for h in hyps]
    try:
        import jiwer

        return {
            "wer": jiwer.wer(refs, hyps),
            "cer": jiwer.cer(refs, hyps),
            "n_words": sum(len(r.split()) for r in refs),
            "n_chars": sum(len(r) for r in refs),
        }
    except ImportError:
        w_err = w_tot = c_err = c_tot = 0
        for r, h in zip(refs, hyps):
            rw, hw = r.split(), h.split()
            w_err += _levenshtein(rw, hw)
            w_tot += len(rw)
            c_err += _levenshtein(list(r), list(h))
            c_tot += len(r)
        return {
            "wer": w_err / max(w_tot, 1),
            "cer": c_err / max(c_tot, 1),
            "n_words": w_tot,
            "n_chars": c_tot,
        }


def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
