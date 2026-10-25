You are auditing a LoRA-finetune pipeline for Higgs Audio v2 (3B) before its results are trusted. Work in
/Users/preyansh/TREX/auto-avsr-realtime/tts_finetune/ (mirrored on the cluster at /scratch/pa2753/higgs_ft/). Read README.md
there first, then higgs_ft_lib.py, gate2_diag.py, phase0_gates.py, train_lora.py, gen_ft.py.

## Context
The finetune trains on the exact inference prompt (system, reference text, reference audio, target text) plus the
target clip's audio tokens as assistant labels (delay pattern, BOS label masked, EOS label kept). Phase 0 "gate 2" was
meant to prove the labels are aligned. Its literal criterion ("matched CE clearly < shuffled-audio CE") FAILED:
4.417 vs 4.463 nats/token on 10 clips. I accepted gate 2 on other evidence: exact parity with the stock HF processor
and HF loss, and labels shifted +-1 frame giving CE 7.0 / 7.6 vs 4.42 aligned. A 40-clip paired diagnostic
(gate2_diag.py) showed wrong target text raises CE in 98% of clips (+0.029 +- 0.003; EOS +0.22) while swapped
same-speaker audio changes nothing (-0.007 +- 0.04), so the shuffled-audio control is weak (same speaker, teacher forcing
is dominated by the audio prefix). Do not repeat those checks. I want STRICTER tests that cannot be passed by a
misaligned pipeline. Run A_r16 (job 19260699, /scratch/pa2753/higgs_ft/runs/A_r16, ~25 min) is already training; do not
touch that run, its lock dir logs/lock_A_r16 or its logs.

## Tests (write new scripts in tts_finetune/strict_gate2/; do not edit existing files)
Pre-register every pass threshold in strict_gate2/THRESHOLDS.md BEFORE running, justified from the noise floor
(bf16 noise from T1 for T1; >= 5x SEM and >= 90% of clips for effect-size tests). Report FAIL plainly; do not retune a
threshold after seeing a result.

T1. Training-path vs inference-path logit parity on the model's own samples (strictest alignment test, no speaker
    confound). With adapter "none" and ref ft_stitch5 (refs/refs_ft.json), generate ~8 tts_val sentences through the
    real inference path (gen_ft.py / gen_higgs Client). Capture the raw audio-head logits at every generation step
    (hook audio_lm_head output, before RAS / top-k / top-p) and the sampled frames. Rebuild the same sequence with
    higgs_ft_lib.build_example (revert the delay pattern on the sampled frames to get (8,T) codes) and run
    forward_logits. Compare per frame and codebook: max/mean |delta log-softmax|, argmax agreement, and log-prob of the
    sampled token. Controls: same comparison with the training sequence shifted by +-1 frame and with a different
    ref clip; both must fail clearly. Include the BOS frame and the EOS step.

T2. Overfit-then-free-run (end-to-end). Train a throwaway LoRA with train_lora.py's exact config (r16, alpha32,
    lr 1e-4, same regex, fp32 LoRA over bf16 base) on K=4 tts_train clips with ONE fixed reference, ~80-150 steps,
    until teacher-forced CE < 0.5 nats. Then free-run generate those 4 targets with the SAME prompt through gen_ft.py
    with the adapter (temperature ~0.3 or greedy-ish, fixed seed), and measure: ASR transcript vs target text
    (use the nemo_asr env; whisper_asr segfaults on some nodes), duration ratio, and token-match rate against the true
    codes. Negative control: repeat with labels deliberately shifted by one frame; it must fail at inference (garbled /
    wrong length) even though its training loss also falls. Also run with eos_labels False and report whether stopping
    still works.

T3. Strong conditioning controls (replace the weak shuffled-audio control). Paired per-clip deltas vs matched, on 40
    tts_train clips, reported for all / cb0 / first 20 frames / EOS:
    (a) target audio from a DIFFERENT speaker (e.g. a TORGO control speaker's clips; read-only access to
        /scratch/th3482/TORGO is fine) with the patient text and patient reference;
    (b) patient target audio but a different-speaker reference (audio + its transcript);
    (c) text swapped for a length-matched but content-different clip (rules out the length confound of the earlier
        shortest<->longest pairing).
    Expect (a) and (b) to be large positive; if (b) is ~0 the reference conditioning is not being used and that is a
    finding to report, not a bug in your test.

T4. Frame-level audit. Decode (audio tokenizer) the sequence reconstructed from build_example's audio_labels via
    revert_delay_pattern and confirm it is bit-identical to decoding the cached codes (data/codes/<utt>.pt). Check
    which frame carries the first and last label, and that under teacher forcing the EOS label sits one frame after the
    last speech frame, matching what generation produces (from T1's samples).

## Deliverables
- strict_gate2/REPORT.md: per test PASS/FAIL with numbers, thresholds quoted from THRESHOLDS.md, 3-sentence verdict, and
  an explicit recommendation: is A_r16 safe to keep (yes / kill and fix, naming the exact bug and line)?
- Scripts + sbatch files + raw JSON outputs in strict_gate2/ (and the cluster mirror).
- Budget: <= ~2 GPU-hours total. Do not start any other training run on the cluster beyond T2's throwaway.

## Rules
- Never write under /scratch/th3482 or /home/th3482; data_june2026_* is off limits; outputs under /scratch/pa2753/.
- No heavy work on the login node; use sbatch/srun on GPU nodes (ssh alias `torch`). Try to use the GPU as much as
  possible and spread jobs across the five accounts (torch_pr_39_general, torch_pr_39_tandon_advanced,
  torch_pr_60_general, torch_pr_60_tandon_advanced, torch_pr_60_tandon_priority). Pick the account/partition that
  starts fastest; hedge-submit to 2-3 pairs and scancel the losers. `sbatch --test-only` estimates are unreliable
  (a100_tandon often pends on QOSGrpGRES; l40s_public with torch_pr_60_general started in ~3 min today). The model
  needs ~15 GiB so an L40S is enough. Request <= 30 min per job.
- Env: python /scratch/pa2753/envs/nemo_asr/bin/python, PYTHONPATH=/scratch/pa2753/higgs_site,
  HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1. Copy gate2_diag.sbatch as a template.
- Report numbers as measured; if a test is inconclusive or you could not run it, say so.
