"""higgs_gen/gen_higgs.py with optional LoRA adapters.

    python gen_ft.py --plan plan.json --task 0 --num-tasks 1 [--limit-units N] [--limit-prompts N]

A work unit may carry "adapter": <dir saved by train_lora.py>. "none"/absent = the pretrained model, untouched
(no peft wrapper at all unless some other unit needs one, in which case the adapter is disabled for the unit).
Adapters are loaded once, kept side by side and switched per unit, so one job can sweep many epochs. Prompt
format, sampling, resumability, output layout and metadata rows are gen_higgs.py's, unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("HIGGS_GEN", "/scratch/pa2753/higgs_gen"))
from gen_higgs import Client, run_unit  # noqa: E402


class FTClient(Client):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.base = self.model
        self.peft = None
        self.loaded: set[str] = set()

    def activate(self, path: str) -> None:
        from peft import PeftModel

        name = "a" + hashlib.sha1(path.encode()).hexdigest()[:8]
        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.base, path, adapter_name=name)
            self.loaded.add(name)
        elif name not in self.loaded:
            self.peft.load_adapter(path, adapter_name=name)
            self.loaded.add(name)
        self.peft.set_adapter(name)
        self.peft.eval()
        self.model = self.peft

    def run(self, unit: dict) -> tuple[int, int]:
        adapter = unit.get("adapter")
        if adapter in (None, "", "none"):
            if self.peft is None:
                return run_unit(self, unit)
            self.model = self.peft
            with self.peft.disable_adapter():
                return run_unit(self, unit)
        self.activate(adapter)
        return run_unit(self, unit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--num-tasks", type=int, required=True)
    ap.add_argument("--limit-units", type=int, default=0)
    ap.add_argument("--limit-prompts", type=int, default=0)
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    mine = plan["units"][args.task :: args.num_tasks]
    if args.limit_units:
        mine = mine[: args.limit_units]
    # adapters are loaded lazily; sort so units sharing an adapter run back to back
    mine.sort(key=lambda u: str(u.get("adapter")))
    total = sum(len(u["prompts"]) for u in mine)
    print(f"task {args.task}/{args.num_tasks}: {len(mine)} units, {total} clips", flush=True)

    client = FTClient(plan["model_path"])
    t0 = time.time()
    done = failed = 0
    for i, unit in enumerate(mine, 1):
        if args.limit_prompts:
            unit = {**unit, "prompts": unit["prompts"][: args.limit_prompts]}
        ok, bad = client.run(unit)
        done += ok
        failed += bad
        print(f"  unit {i}/{len(mine)} {unit['name']} [{unit.get('adapter', 'none')}]: ok={ok} failed={bad}  "
              f"[{(time.time() - t0) / 60:.1f} min elapsed]", flush=True)
    print(f"TASK_DONE ok={done} failed={failed}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
