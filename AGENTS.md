# Repository Guidelines

## Project Structure & Module Organization
TacoRank is a Python scaffold for the RankForge recommender-system harness plus the KuaiRand-Pure starter kit. Core package code lives in `src/tacorank/`, with subpackages for orchestration, research, coding, safety, execution, evaluation, recovery, reporting, memory, and SRE. Dataset-specific integration points live in `benchmarks/kuairand_pure/`.

Top-level benchmark scripts include `data.py`, `evaluate.py`, `baseline.py`, `submit.py`, and `ablation_features.py`. Put candidate model work under `solution/`. Generated outputs belong in `runs/` and `artifacts/`. Downloaded data belongs in git-ignored `KuaiRand-Pure/data/`.

## Build, Test, and Development Commands
- `python3 baseline.py --model random`: run the random baseline as an evaluator sanity check.
- `python3 baseline.py --model fm`: train and evaluate the official FM baseline. Add `--data_dir` for non-default data paths.
- `python3 submit.py --make --split test submission.csv`: generate a sample FM submission.
- `python3 submit.py --check --split test submission.csv`: validate submission schema, row order, and finite scores.
- `python3 submit.py --score --split valid submission.csv`: score a valid split submission locally.
- `PYTHONPATH=src python3 -m pytest`: run the repository tests once pytest is installed.

## Coding Style & Naming Conventions
Use Python 3.9+ and keep dependencies minimal; the starter kit currently requires only `numpy`. Follow PEP 8 with 4-space indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Prefer small modules that map to existing subpackage responsibilities. Keep `evaluate.py` behavior stable because it defines the competition contract.

## Testing Guidelines
Tests are organized under `tests/<area>/`, matching `src/tacorank/<area>/`. Name files `test_<module>.py` and functions `test_<behavior>()`. Add unit tests for pure logic and integration tests for orchestration, execution, adapters, and submission paths. For metric or submission changes, include fixtures covering ties, zero-positive users, duplicate `(user_id, video_id)` pairs, and invalid scores.

## Commit & Pull Request Guidelines
Existing history uses short imperative commit subjects, for example `Add official KuaiRand starter kit`. Keep subjects concise and repository-visible. Pull requests should include motivation, touched areas, commands run, data assumptions, and metric deltas. Link related issues when available and include screenshots only for generated reports or visual artifacts.

## Security & Configuration Tips
Do not commit downloaded KuaiRand data, archives, secrets, run ledgers with sensitive paths, or large generated artifacts. Treat `contract/`, `evaluate.py`, and submission validation rules as protected surfaces unless a task explicitly changes the benchmark contract.
