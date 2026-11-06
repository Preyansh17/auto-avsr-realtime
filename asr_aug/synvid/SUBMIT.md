# synvid external-setup jobs (CPU, partition cs, account torch_pr_39_general)

Run from the Mac. Each job writes its log to /scratch/pa2753/synvid/logs/<job-name>_<jobid>.out.
All four are CPU-only and independent; usr2 is the one Phase 1.1–1.4 needs first.

| order | job | what | pinned |
|---|---|---|---|
| 1 | build_env_usr2.sbatch | env usr2 (py3.10, torch 2.4.1 cu121, usr2 deps, ibug RetinaFace+FAN) + USR 2.0 HR Large/Huge VSR checkpoints | usr2 e9e9c94, face_detection db2a4e8, face_alignment 9cf5494 |
| 2 | build_env_keysync.sbatch | env keysync (py3.11, torch 2.4.1 cu121) + HF weights -> models/keysync | code 5e2503b, HF c26cc46 |
| 3 | build_env_latentsync.sbatch | env latentsync (py3.10.13, torch 2.5.1 cu121) + HF weights -> models/latentsync (skip if KeySync only) | code a229c39, HF c42c7e6 |
| 4 | fetch_text.sbatch | librispeech-lm-norm.txt.gz (openslr 11) -> data/ | — |

One at a time:

```bash
ssh torch sbatch --account=torch_pr_39_general --partition=cs /scratch/pa2753/synvid/code/build_env_usr2.sbatch
```
```bash
ssh torch sbatch --account=torch_pr_39_general --partition=cs /scratch/pa2753/synvid/code/build_env_keysync.sbatch
```
```bash
ssh torch sbatch --account=torch_pr_39_general --partition=cs /scratch/pa2753/synvid/code/build_env_latentsync.sbatch
```
```bash
ssh torch sbatch --account=torch_pr_39_general --partition=cs /scratch/pa2753/synvid/code/fetch_text.sbatch
```

All four, in order:

```bash
ssh torch 'cd /scratch/pa2753/synvid/code && for j in build_env_usr2 build_env_keysync build_env_latentsync fetch_text; do sbatch --account=torch_pr_39_general --partition=cs $j.sbatch; done'
```

KeySync only (no LatentSync):

```bash
ssh torch 'cd /scratch/pa2753/synvid/code && for j in build_env_usr2 build_env_keysync fetch_text; do sbatch --account=torch_pr_39_general --partition=cs $j.sbatch; done'
```

Check afterwards:

```bash
ssh torch 'sacct -u pa2753 -S today -n -X -o JobID,JobName%20,State,Elapsed | grep sv_; tail -n 5 /scratch/pa2753/synvid/logs/sv_*.out'
```
