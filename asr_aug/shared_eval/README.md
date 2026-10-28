# Shared-split eval

Score any system on identical rows and references from the shared role files.

- Rows and refs: Tianyu's split tree, `LATEST_combined/roles/<role>.jsonl` (combined_v6 as of 2026-10-25).
  The scorer prints the resolved path and sha256 so a result records which version it used.
- Scorer: `score_shared.py` (copy on torch: `/scratch/pa2753/shared_eval/`). Plain python3, no deps.
- Input: a hyps TSV with a header, a `hyp` column, and `take_id` or `path` (file stem = take_id). Refs in the TSV are ignored.
- One TSV per seed; pass all seeds of a system together to get mean ± sd.

```bash
python3 score_shared.py --role s1_test --label medium_balanced_hybrid --tsv run_seed*/s1_test_hyps.tsv
```

## Our systems (2026-10-26, `/scratch/pa2753/shared_eval/ours_table.txt`), WER %, mean ± sd over seeds

| system | seeds | s1_val | s1_test | s1_oov | s2_val | s2_test | s2_oov |
|---|---|---|---|---|---|---|---|
| Whisper large-v3 zero-shot | 1 | 98.4 | 85.2 | 94.6 | 37.4 | 28.4 | 35.1 |
| large-v3 FT, Oct 1 s1 train (N) | 10 | 2.5 ± 0.7 | 7.0 ± 2.5 | 30.7 ± 2.2 | | | |
| large-v3 FT, old + Oct 1 (ON) | 10 | 2.6 ± 0.6 | 7.3 ± 2.0 | 29.7 ± 5.0 | | | |
| ON + Higgs synthetic (ONS) | 10 | 2.9 ± 0.7 | 8.3 ± 2.2 | 22.0 ± 2.8 | | | |
| large-v3 FT s1 (multi-patient lane) | 3 | 2.2 ± 0.6 | 8.3 ± 0.6 | 29.3 ± 4.5 | | | |
| large-v3 FT s2, lang_task prefix | 3 | | | | 0.6 | 1.0 ± 0.7 | 6.0 ± 2.3 |
| large-v3 FT s2, legacy prefix | 3 | | | | 0.6 | 0.6 | 5.7 ± 1.5 |
| large-v3 FT pooled s1+s2 | 3 | | | | 0.6 | 1.2 ± 0.6 | 5.9 ± 0.9 |
