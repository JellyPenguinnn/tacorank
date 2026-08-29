# Repository Guidelines

## Project Structure & Ownership
TacoRank is a Python harness for automated recommender-system research. Core code lives in `src/tacorank/`: planning and research, orchestration, context, coding, safety, execution, memory, recovery/SRE, evaluation, reflection, and reporting are separate subsystems. Keep changes within the subsystem that owns the behavior. Shared event contracts belong in `src/tacorank/schemas.py`; the memory layer is the sole owner of persisted event-ledger writes.

KuaiRand-specific adapters live in `benchmarks/kuairand_pure/`. The official starter kit is the `kuairand-starter-kit/` Git submodule. Tests mirror package areas under `tests/`. Put candidate solutions in `solution/`, reusable research material in `research/methods/`, and generated output in ignored `runs/` or `artifacts/`. Shared architecture requirements are documented in `TacoRank-Memory-Schema-v1.md` and `docs/`; personal planning notes remain local.

## Setup and Development Commands
- `git submodule update --init --recursive`: fetch the pinned starter kit.
- `python3 -m venv .venv`: create the repository-local Python environment.
- `.venv/bin/python -m pip install -r requirements-dev.txt`: install runtime dependencies and pytest.
- `.venv/bin/python -m pip install -e .`: install the `tacorank` CLI in editable mode.
- `PYTHONPYCACHEPREFIX=/tmp/tacorank-pycache .venv/bin/python -m pytest`: run the complete test suite.
- `PYTHONPATH=src:. python3 -m unittest discover -s tests -p 'test_*.py' -v`: run unittest-compatible tests when pytest is unavailable; this does not cover every pytest test.
- `cd kuairand-starter-kit && python3 baseline.py --data_dir ../KuaiRand-Pure/data --model random`: sanity-check the evaluator.
- `cd kuairand-starter-kit && python3 baseline.py --data_dir ../KuaiRand-Pure/data --model fm --seed 0`: reproduce the FM baseline.
- From the starter-kit directory, run `python3 submit.py --check --split test submission.csv` before submission.

## Harness Operations
The contract and protected-path files must be frozen, all placeholder hashes in copies of `config.example.json` and `live-adapters.example.json` must be replaced, and the live worktree/data/runtime prerequisites must pass preflight before starting a production run.

- `tacorank run --config run-config.json --live-config live-adapters.json`: start one production experiment using the hash-bound live adapters. The command fails closed before creating ledger state when prerequisites are missing.
- `tacorank run --config test-config.json --allow-test-adapters`: run deterministic fake adapters only when the config explicitly sets `adapter_mode` to `fake`; this mode is for tests, not benchmark evidence.
- `tacorank resume --run-id run_001 --repository-root .`: repair a truncated ledger tail if necessary and print the phase from which orchestration can be resumed; it does not yet restart adapter execution.
- `tacorank status --run-id run_001 --repository-root .`: print the current projected run state.
- `tacorank validate-ledger --run-id run_001 --repository-root .`: validate schema, sequence, hashes, transitions, and referenced artifacts.
- `tacorank rebuild-views --run-id run_001 --repository-root .`: regenerate `STATUS.md`, `LESSONS.md`, and `SUMMARY.md` from the ledger.
- `tacorank finalize --run-id run_001 --repository-root .`: validate that the run is stopped, then fail closed because the standalone reproduction/final-selection command is not implemented yet.

## Coding Style & Naming
Support Python 3.9+. Follow PEP 8 with four-space indentation, `snake_case` modules/functions, `PascalCase` classes, and uppercase constants. Add type hints to public APIs and prefer immutable dataclasses or Pydantic models for structured state. Keep decision, trust, and metric logic deterministic; avoid hidden global state and unnecessary dependencies.

## Evaluation Contract & Safety
Treat `kuairand-starter-kit/evaluate.py`, data splits, label semantics, submission ordering, and contract hashes as protected benchmark surfaces. Evaluation is within-user ranking on `long_view`; primary is `mean(GAUC, nDCG@5)`. Preserve zero-positive-user and GAUC weighting rules exactly. Proxy and unbiased-audit results must not update the public-validation best score; hidden-final results never feed planning. Do not commit data, secrets, submissions, model artifacts, or sensitive run ledgers.

## Testing Guidelines
Use pytest as the complete test runner; standard-library `unittest.TestCase` tests remain supported. Name files `test_<module>.py` and tests `test_<behavior>`. Cover metric ties, single-class users, non-finite predictions, duplicate `(user_id, video_id)` rows, row-order violations, trust ordering, and deterministic seeds. Run baseline parity checks after adapter or metric changes.

## Commits & Pull Requests
Use concise imperative subjects, optionally with the existing `[fea]:` prefix. Keep submodule pointer changes explicit. Pull requests should state motivation, affected subsystem, commands run, data/split assumptions, seed values, and metric deltas; link issues and attach report screenshots only when relevant.

# Pull Request Review Workflow

When asked to address PR feedback, do NOT blindly implement reviewer comments.

For every review comment:

1. Retrieve the full PR context, including:
   - PR description
   - review comments
   - inline code comments
   - relevant commits and diffs

2. Inspect the surrounding implementation before making changes.
   Understand:
   - what the current code is trying to achieve
   - why it may have been implemented this way
   - where this code sits in the overall execution/data flow
   - its callers, downstream consumers, interfaces, invariants, and tests

3. Interpret the reviewer's suggestion.
   Identify:
   - what problem the reviewer is trying to prevent or solve
   - what assumptions their suggestion makes
   - how their proposed logic would change the existing flow

4. Compare the two approaches rather than assuming the reviewer is correct.

   Evaluate both against:
   - intended application flow
   - product/business requirements
   - existing architecture
   - correctness and edge cases
   - consistency with related code
   - backward compatibility
   - tests
   - maintainability and unnecessary complexity

5. Classify each comment as one of:
   - ACCEPT: reviewer logic is more correct
   - PARTIAL: reviewer identified a valid issue, but the suggested solution is not ideal
   - KEEP CURRENT: current implementation is correct and reviewer suggestion would break or weaken the intended flow
   - NEEDS CONTEXT: repository evidence is insufficient to determine the correct behavior

6. Before modifying code, explain:
   - current logic
   - reviewer logic
   - relevant execution/data flow
   - trade-offs
   - your conclusion
   - evidence supporting the conclusion

7. If the reviewer is correct, implement the change.

8. If the reviewer identified a real problem but proposed the wrong fix, implement a better fix and explain why.

9. If the current implementation is correct, do not change it merely to satisfy the comment. Explain clearly why the existing behavior should remain.

10. Check for semantic consequences outside the commented lines. A locally reasonable change may break another part of the system.

11. Run relevant tests, linting, and type checks after changes.

Never commit, push, merge, force-push, or resolve/close review conversations without explicit approval.
