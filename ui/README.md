# TacoRank Run Dashboard

Local repository-backed dashboard for starting and inspecting TacoRank runs.
It lists every ledger under `runs/`, opens the latest run by default, refreshes
every five seconds, and shows each iteration's plan, memory/context, patch,
Gate A, execution, Gate B, evaluation, recovery, and raw ledger evidence.

## Run locally

```bash
cd ui
npm run dev
```

Open the printed loopback URL. Set `TACORANK_REPOSITORY_ROOT` only when the UI
directory is not directly inside the repository. To enable **Start new run**,
export `DEEPSEEK_API_KEY` in the same shell before starting the dashboard. The
key is checked for presence only and is never returned to the browser.

The start action asks for confirmation and launches the repository's reviewed
`run-new-live.sh` workflow. That is a paid production workflow and includes
setup, preflight, autonomous execution, finalization, and validation. Launcher
logs are written under ignored `.tacorank/dashboard-launches/`.

## Validate

```bash
npm run lint
npm run build
```
