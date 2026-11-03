# My first Blume delivery

## Outcome
`demo_realtime.py` fails fast with a one-line, clear error when `--video`, `--checkpoint`, `--jit-model` or `--sp-model-path` points to a missing file, instead of a traceback from deep inside model load or video decode.

## Acceptance checks
- A missing `--video` exits non-zero with a message naming the path, before any model loads — unit: tests/test_smoke.py `#missing-video`
- A missing `--checkpoint`, `--jit-model` or `--sp-model-path` does the same — unit: tests/test_smoke.py `#missing-model-files`
- Valid paths behave exactly as today — unit: tests/test_smoke.py `#no-regression`
- The error text reads clearly on the command line — manual `#error-wording`

## Approach
After `parse_args()` in `demo_realtime.py`, add a small `validate_paths(args)` that checks each supplied path with `os.path.isfile` and calls `parser.error(...)` (exit code 2, usage line shown). Run it before the default JIT asset download so a bad path never triggers a download. Cover it with CPU-only tests in `tests/test_smoke.py`.

## Decisions
- Use `parser.error` rather than a custom exception, to match argparse conventions already used by the CLI.
- Validate only paths the user supplied; do not check the auto-downloaded default assets.
- Plan only for now: no project files other than this plan are changed until the person approves Build.

## Tasks
### Plan
- [x] Find the improvement and write this plan `#write-plan`
### Build
- [ ] Add `validate_paths` to `demo_realtime.py` and call it after argument parsing `#validate-paths`
- [ ] Add tests for missing and valid paths `#add-tests`
### Test
- [ ] Run `python -m tests.test_smoke` and record the real result `#run-tests`
### Review
- [ ] Person reviews the outcome `#review`
