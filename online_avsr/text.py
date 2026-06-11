import math
import os
import re
import tempfile
from typing import Iterable, List, Sequence, Tuple

from . import EXPECTED_SPM_VOCAB_SIZE


def normalize_text(text: str) -> List[str]:
    if text is None:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, a_token in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, b_token in enumerate(b, start=1):
            cost = 0 if a_token == b_token else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text(reference)
    hyp_tokens = normalize_text(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return levenshtein_distance(ref_tokens, hyp_tokens) / float(len(ref_tokens))


def extract_reference_from_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    reference = name.split("_")[0].replace("-", " ")
    return re.sub(r"\s+", " ", reference).strip()


def checkpoint_id(path: str, prefix_len: int = 12) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:prefix_len]


def load_sentencepiece_model(sp_model_path: str, expected_size: int = EXPECTED_SPM_VOCAB_SIZE):
    import sentencepiece as spm

    if not sp_model_path or not os.path.isfile(sp_model_path):
        raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")
    sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
    actual_size = sp_model.get_piece_size()
    if actual_size != expected_size:
        raise ValueError(
            f"Expected SentencePiece vocab size {expected_size}, got {actual_size}: {sp_model_path}"
        )
    return sp_model


def post_process_hypotheses(hypotheses, sp_model) -> List[Tuple[str, float, List[int]]]:
    remove_ids = {sp_model.unk_id(), sp_model.eos_id(), sp_model.pad_id()}
    processed = []
    for hyp in hypotheses:
        token_ids = list(hyp[0])[1:]
        filtered = [idx for idx in token_ids if idx not in remove_ids]
        score = math.exp(float(hyp[3]))
        processed.append((sp_model.decode(filtered), score, token_ids))
    return processed


def train_sentencepiece_from_texts(
    texts: Iterable[str],
    model_prefix: str,
    vocab_size: int = EXPECTED_SPM_VOCAB_SIZE,
) -> str:
    import sentencepiece as spm

    os.makedirs(os.path.dirname(os.path.abspath(model_prefix)), exist_ok=True)
    normalized = [line.strip().lower() for line in texts if line and line.strip()]
    if not normalized:
        raise ValueError("Cannot train SentencePiece model without non-empty text")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as corpus:
        corpus_path = corpus.name
        for line in normalized:
            corpus.write(line + "\n")

    try:
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            bos_id=-1,
            pad_id=0,
            eos_id=1,
            unk_id=2,
            hard_vocab_limit=False,
        )
    finally:
        try:
            os.unlink(corpus_path)
        except OSError:
            pass

    return model_prefix + ".model"
