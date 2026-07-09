#!/usr/bin/env python3
"""Read a field (or the whole dict) out of the newest eval.py summary.json
matching a glob pattern. Exists purely to keep slurm/select_best_streaming_epoch.sbatch's
shell quoting sane -- no inline `python -c "..."` blocks nested inside an
already-escaped singularity/bash-lc string.

  python scripts/read_eval_summary.py '<dir>/eval_streaming_*/summary.json' corpus_wer
  python scripts/read_eval_summary.py '<dir>/eval_streaming_*/summary.json'   # whole dict
"""
import glob
import json
import sys

pattern = sys.argv[1]
key = sys.argv[2] if len(sys.argv) > 2 else None

matches = sorted(glob.glob(pattern))
if not matches:
    print(f"[FATAL] no file matched: {pattern}", file=sys.stderr)
    sys.exit(1)

data = json.load(open(matches[-1]))
print(data[key] if key else data)
