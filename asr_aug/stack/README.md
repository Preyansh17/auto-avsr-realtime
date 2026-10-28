# Item 3: stack Tianyu's winning semg1 ingredients, 5 seeds (PLAN.md 3.3)

Pre-registered 2026-10-26, before any adapter, synthetic clip or ASR run of this study existed. Cluster mirror:
`/scratch/pa2753/stack/{code,higgs,data,root,runs,logs,jobs.txt}`.

## Question
Tianyu's semg1 best is test30 4.1 / OOV 10.0 (Notion; `docs/investigations/tianyu_notion_2026-10-26.md`).
It is built from encoder-only training, old recordings, both patients, fine-tuned Higgs synthetic, weight averaging
and voting, each shown separately with 1-3 seeds. Does the stack hold up over 5 seeds, and does adding semg2 plus
synthetic speech help semg1 on top of Tianyu's documented recipe?

## Design (user decisions 2026-10-26)
- Model: Whisper large-v3, **encoder-only** finetune (all 32 encoder layers; decoder, conv stem and positional
  embeddings frozen), lr 1e-5, 30 epochs, warmup 25, wd 0.01, batch 8 x 2, bf16, SpecAugment defaults, lang_task
  prefix. Checkpoint = best s1_val WER (secondary: last epoch). `whisper_stack.sbatch`.
- Arms (`make_mix.py`), seeds 1-5:
  - **E1** = old split_v1 merged train (314) + semg1 Oct 1 train (240).
  - **E2** = E1 + semg2 train (240) + synthetic at 0.5 x real per voice.
    - semg1 voice (277 clips): half from the ONS pool (old-data adapter, `asr_aug/data/synth_kept_{A,B}`), half from
      a new Oct 1 adapter.
    - semg2 voice (120 clips): a new semg2 adapter.
    - Each voice's share is half pool A, half pool B.
- New adapters (`prep_tts.py`, `tts_train.sbatch`): Higgs B_r64, epoch 8 fixed, trained on that patient's train role
  only. Reference = ft_stitch5 from train clips. They speak the pool A/B text minus any s1/s2 val/test/oov sentence
  (`make_gen_plan.py`).
- Filter (`filter_synth.py`): the H rules and H thresholds, frozen (pool A WER 0.576 / insertions 1; pool B CER 0.521).
  The judge is that patient's real-only large-v3. Thresholds are not re-derived because these judges have 1-3% val WER.
- Ensembles per arm: **soup** = uniform weight average of the 5 seeds' best/ (`soup.py`). **MBR vote** = per clip, the
  seed hypothesis with the lowest summed word edit distance to the others (`report_stack.py`).

## Analysis (`report_stack.py`)
- **Primary:** E2 - E1 on s1_oov WER, single seeds, best checkpoint. Paired two-level bootstrap (seeds, then
  sentences; 4000 draws). "E2 helps" = 95% CI upper bound < 0.
- **Secondary** (CIs reported, no claims):
  - E2 - E1 on s1_test, s2_test and s2_oov.
  - Soup and MBR vs the single-seed mean, per arm, on s1_oov and s1_test.
  - All of the above on the last-epoch checkpoint.
- **Bar:** the best system vs Tianyu's 4.1 / 10.0, descriptive only (different seeds, and Tianyu's runs are single).
- Item 1 (contextual biasing) is then built on the E-arm checkpoints: the soup, plus single seeds for variance.

## Known limits
- 30 test / 49 OOV sentences: the seed sd on OOV was 2-5 pp in H, so effects under ~4 pp are unlikely to resolve with 5 seeds.
- E2 changes two things at once (semg2 real + synthetic); this was chosen deliberately (2 arms, not 4) and cannot
  attribute the effect.
- Pool text is old-vocabulary (H caveat): 67 of 94 OOV words are covered by the pools.

## Item 1: contextual biasing (pre-registered 2026-10-26, before any biased decode)
Run on each arm's soup and on its 5 single seeds (`bias.sbatch`), so seed variance is available.
- **Methods** (beam 10, lang_task prefix; `bias_decode.py`, `llm_rescore.py`):
  - beam: plain beam search, the reference.
  - prompt: bias words as the Whisper prompt.
  - ngram: shallow fusion with an interpolated token 4-gram of the bias text, + lam x log P_lm per step.
  - llm: Qwen2.5-1.5B rescoring of the beam 10-best, + mu x mean token log-prob, with a prefix naming the bias words.
- **Bias text** (`bias_text.py`), two conditions reported apart:
  - **clean (the claim):** only text the models already train on (s1/s2 train, old train, pool A/B; prompt words =
    the s1/s2 training vocabulary).
  - **words (ceiling):** clean + the s1/s2 OOV words as a bare list, i.e. "a caregiver supplies the vocabulary".
    The test words are known in this condition, so it is never the claim.
- **Selection:** lam in {0.1, 0.2, 0.3, 0.5} and mu in {0.1, 0.25, 0.5, 1.0}, picked by lowest pooled s1_val + s2_val
  WER. val has no OOV words, so this is conservative. The other grid values are decoded on OOV, descriptive only.
- **Primary:** each clean method minus beam on s1_oov, 5 single seeds per arm. Paired two-level bootstrap; Holm over
  the 3 methods. "Helps" = Holm-adjusted p < 0.05 and CI upper < 0, and the s1_test point estimate of the same
  difference is at most +2 pp.
- **Secondary:** the same on s2_oov; the words condition; the soup.
- **Bar:** Tianyu semg1 OOV 10.0, semg2 OOV 2.1.

## Log
- 2026-10-26: both adapters trained (epoch 8). Generated 1,998 clips per voice (2 pool sentences dropped as held text).
- Filter (`/scratch/pa2753/stack/data/filter_report.md`), kept: s1 Oct 1 voice A 314/1000, B 451/998; s2 voice A 815/1000, B 936/998.
  Almost all rejections come from the judge. **Caveat:** the s1 judge is trained only on the 50-word Oct 1 set, while H's
  judge knew the old vocabulary of pool A. So s1 rejection mixes audio quality with judge vocabulary, and the kept s1
  clips lean toward easier sentences. Thresholds stayed frozen as pre-registered. E2 needs 69/70 (s1) and 60/60 (s2), so supply is fine.
- E2 seed 1 mix: 1,191 rows (old 314, semg1 240, semg2 240, synthetic 397 = 138 old-adapter + 139 Oct 1 adapter + 120 semg2).
- bias_decode: Whisper's generate() with num_return_sequences=10 ran out of memory (it expands the batch by both beams and return
  sequences). Switched to GenerationMixin.generate with an explicit prefix (16.6 GB for a 10-best on an L40S).
- 2026-10-27: bias job 20971197 failed in n-gram construction: `generation_config.eos_token_id` is a list, so it was
  appended as a token. Now `<|endoftext|>` (50257). Tested on a CPU node (srun 21158524): both LMs normalise to 1.0.
  Beam/prompt val decodes had already run. Resubmitted bias for 9 trained seeds (21158680-689) and the E1 soup (21158690).

**Amendment 1 to item 1 (2026-10-28, written before any test or OOV biasing result was read).** On the first finished
run (E1_seed4), LLM rescoring was worse than plain beam on val at every pre-registered mu (pooled s1/s2 val mean:
beam 8.24, mu 0.1 10.44, 0.25 11.26). Whisper's length-normalised sequence score spans about 0.1 over the 10-best,
while Qwen's mean token log-prob spans several units, so mu >= 0.1 lets the LLM outweigh the acoustics. The mu grid
becomes {0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0}, selected on val as before. Rescoring is re-run from the saved
beam 10-best lists (no new decodes). Everything else is unchanged.
n-gram fusion is unchanged. At lam 0.3 one E1_seed4 val clip looped ("i need to go to the bathroom" repeated, 524
characters), giving val WER 44.5 vs 9.9 at 0.2. Repetition loops are counted and reported per method and set (hyp at
least 2x ref words + 5, or a word 4-gram that occurs 3+ times); no loop guard is added.

## Results (2026-10-28)
Summary and tables: PLAN.md section 3.6. Raw outputs: `results/report_stack.out`, `results/report_bias.out`.
- Item 3 primary passes: E2 - E1 on s1_oov -6.01 pp [-11.62, -0.98] (p 0.020); E2 soup s1 test/OOV 7.10 / 8.07, s2 0.59 / 1.75.
- Item 1: all six primaries fail (no method helps on either arm). The dominant failure is repetition loops (prompt with
  the words list, n-gram at lam >= 0.2, LLM at mu >= 0.05).
- report_bias.py fix before its first full run: `loop()` returned an empty Counter on hyps under 4 words (TypeError).
