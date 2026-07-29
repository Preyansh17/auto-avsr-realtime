#!/usr/bin/env python3
"""Ground-truth word timings from Montreal Forced Aligner TextGrids.

Why this exists
---------------
The streaming eval scripts originally measured latency against the *model's own*
predicted word timestamps:

    word_lag = emit_time - predicted_word_end

Both terms come from the same model, which makes the metric self-referential:
if a checkpoint emits later, its own predicted timestamps shift later too and
the difference partly cancels. MFA gives a fixed external reference instead.

What this actually changed (measured 2026-07-25, Nemotron epochs=210 +
speed-perturbation, 3 seeds x merged/legal/legacy):

  * Median latency got BETTER, not worse -- merged word lag 1.02s -> 0.70s,
    TTFT 0.99s -> 0.80s. The a-priori expectation was the opposite (RNN-T
    emission is delayed, so its predicted word-end should be late, and a late
    subtrahend shrinks the measured lag). It did not hold: MFA marks a word's
    full acoustic extent, and on elongated dysarthric speech the model often
    commits from partial evidence before the word finishes, so the model's own
    anchor sits EARLIER than MFA's, not later.
  * Tails got worse and are the real story -- merged p95 lag 1.36s -> 1.76s,
    p95 TTFT 1.10s -> 1.83s. The self-referential metric was compressing
    exactly the cases a user notices.
  * Most useful: it exposed a systematic ~0.4s difference in emission
    eagerness BETWEEN checkpoints (seed1 TTFT ~1.1-1.25s vs seeds 2-3 at
    0.63-0.85s, consistent across all three domains) that the model-referenced
    metric could not see at all -- it read all three seeds flat at ~0.95-1.16s.

An earlier argument that att_context_size=[70,13] (13 lookahead frames x 0.08s
= 1.04s right context) sets a floor below which measured lag cannot fall was
NOT borne out -- MFA-referenced mean lag is 0.65-0.93s, below it. Do not reuse
that reasoning.

It also matters for cross-model comparison: Whisper's timestamps come from a
different mechanism with a different bias, so a Whisper-vs-Nemotron latency
comparison on model-predicted references is partly comparing timestamp biases
rather than latencies. MFA puts both on one neutral reference.

Semantics
---------
Deliberately keeps the *definition* of the metric unchanged (anchor = the time
the word finished being spoken) and swaps only the *source* of that anchor, so
old and new numbers stay directly comparable.

Only words the model got right are timed. A substituted or hallucinated word has
no meaningful acoustic onset to measure against, so it is excluded rather than
matched to an arbitrary neighbour.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from asr_baselines.metrics import normalize

# The "words" IntervalTier of a long-format MFA TextGrid: every interval is
# (xmin, xmax, text); empty text == silence. Parsed here rather than imported
# from patient_audio.py (which has the same logic) so this module stays
# dependency-light: the Whisper eval runs from a packaged tree that ships only
# a subset of asr_baselines, and patient_audio pulls in torch, which nothing
# here needs.
_TG_INTERVAL = re.compile(
    r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"([^"]*)"'
)


def parse_textgrid_words(path: str) -> Optional[List[Tuple[str, float]]]:
    """Return [(word, end_sec), ...] for the 'words' tier, or None if unparseable."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'name\s*=\s*"words".*?(?=\n\s*item\s*\[|\Z)', content, re.DOTALL)
    if not m:
        return None
    words = []
    for im in _TG_INTERVAL.finditer(m.group(0)):
        text = im.group(3).strip()
        if text:
            words.append((text, float(im.group(2))))
    return words or None


def textgrid_name_for(audio_filepath: str, split_root: str) -> str:
    """MFA corpus names flatten the split-relative path: a/b/c.wav -> a__b__c.TextGrid.

    (make_carelesswhisper_dataset.py builds the corpus that way so utterance
    names are unique and reversible.)
    """
    rel = audio_filepath
    if split_root and rel.startswith(split_root):
        rel = rel[len(split_root):]
    rel = rel.lstrip("/")
    return rel.replace("/", "__").rsplit(".", 1)[0] + ".TextGrid"


def build_textgrid_index(aligned_root: str) -> Dict[str, str]:
    """basename -> full path, over every *.TextGrid under aligned_root."""
    index = {}
    for dirpath, _dirnames, filenames in os.walk(aligned_root):
        for fn in filenames:
            if fn.endswith(".TextGrid"):
                # First writer wins; duplicates across split subdirs are the
                # same utterance aligned once per split membership.
                index.setdefault(fn, os.path.join(dirpath, fn))
    return index


def ref_word_ends(audio_filepath: str, index: Dict[str, str],
                  split_root: str) -> Optional[List[Tuple[str, float]]]:
    """[(normalized_word, end_sec), ...] for one clip, or None if unaligned."""
    tg = index.get(textgrid_name_for(audio_filepath, split_root))
    if tg is None:
        return None
    words = parse_textgrid_words(tg)
    if not words:
        return None
    out = []
    for w, end in words:
        n = normalize(w)
        if n:
            # normalize() may split a token; MFA words are single tokens in
            # practice, but keep the first piece rather than silently dropping.
            out.append((n.split()[0], float(end)))
    return out or None


def match_hyp_to_ref(hyp_words: List[str],
                     ref_words: List[str]) -> List[Tuple[int, int]]:
    """Levenshtein alignment; returns (hyp_idx, ref_idx) for EXACT matches only.

    Substitutions/insertions/deletions are dropped -- a wrong word has no
    reference onset worth measuring against.
    """
    n, m = len(hyp_words), len(ref_words)
    # dp[i][j] = edit distance between hyp[:i] and ref[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hyp_words[i - 1] == ref_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if hyp_words[i - 1] == ref_words[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            if cost == 0:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs
