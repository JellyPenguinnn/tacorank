# Repository Guidelines

## Project Structure & Ownership
TacoRank is a Python harness for automated recommender-system research. Core code lives in `src/tacorank/`: planning and research, orchestration, context, coding, safety, execution, memory, recovery/SRE, evaluation, reflection, and reporting are separate subsystems. Keep changes within the subsystem that owns the behavior. Shared event contracts belong in `src/tacorank/schemas.py`; the memory layer is the sole owner of persisted event-ledger writes.

KuaiRand-specific adapters live in `benchmarks/kuairand_pure/`. The official starter kit is the `kuairand-starter-kit/` Git submodule. Tests mirror package areas under `tests/`. Put candidate solutions in `solution/`, reusable research material in `research/methods/`, and generated output in ignored `runs/` or `artifacts/`. Architecture and role requirements are documented in `TacoRank-Memory-Schema-v1.md` and `eesyuen-agents/`.

## Setup and Development Commands
- `git submodule update --init --recursive`: fetch the pinned starter kit.
- `python3 -m pip install -r requirements.txt`: install NumPy, pandas, and Pydantic.
- `PYTHONPATH=src:. python3 -m unittest discover -s tests -p 'test_*.py' -v`: run the complete test suite.
- `cd kuairand-starter-kit && python3 baseline.py --data_dir ../KuaiRand-Pure/data --model random`: sanity-check the evaluator.
- `cd kuairand-starter-kit && python3 baseline.py --data_dir ../KuaiRand-Pure/data --model fm --seed 0`: reproduce the FM baseline.
- From the starter-kit directory, run `python3 submit.py --check --split test submission.csv` before submission.

## Coding Style & Naming
Support Python 3.9+. Follow PEP 8 with four-space indentation, `snake_case` modules/functions, `PascalCase` classes, and uppercase constants. Add type hints to public APIs and prefer immutable dataclasses or Pydantic models for structured state. Keep decision, trust, and metric logic deterministic; avoid hidden global state and unnecessary dependencies.

## Evaluation Contract & Safety
Treat `kuairand-starter-kit/evaluate.py`, data splits, label semantics, submission ordering, and contract hashes as protected benchmark surfaces. Evaluation is within-user ranking on `long_view`; primary is `mean(GAUC, nDCG@5)`. Preserve zero-positive-user and GAUC weighting rules exactly. Proxy and unbiased-audit results must not update the public-validation best score; hidden-final results never feed planning. Do not commit data, secrets, submissions, model artifacts, or sensitive run ledgers.

## Testing Guidelines
Use standard-library `unittest`; name files `test_<module>.py` and tests `test_<behavior>`. Cover metric ties, single-class users, non-finite predictions, duplicate `(user_id, video_id)` rows, row-order violations, trust ordering, and deterministic seeds. Run baseline parity checks after adapter or metric changes.

## Commits & Pull Requests
Use concise imperative subjects, optionally with the existing `[fea]:` prefix. Keep submodule pointer changes explicit. Pull requests should state motivation, affected subsystem, commands run, data/split assumptions, seed values, and metric deltas; link issues and attach report screenshots only when relevant.
