# TacoRank Run Dashboard

Local repository-backed dashboard for starting and inspecting TacoRank runs.
It lists every ledger under `runs/`, opens the latest run by default, refreshes
every five seconds, and shows each iteration's plan, memory/context, patch,
Gate A, execution, Gate B, evaluation, recovery, and raw ledger evidence. The
current-run panel updates elapsed stage/iteration time and last-event age every
second from ledger timestamp anchors, and shows the active attempt, fidelity,
timeout, deadline, and loop step.

## Run locally

```bash
cd ui
npm run dev
```

Open the printed loopback URL. Set `TACORANK_REPOSITORY_ROOT` only when the UI
directory is not directly inside the repository. **Start new run** prompts for
the DeepSeek API key in a masked field. The key is sent only to the local start
endpoint, passed to that run's launcher environment, and is not saved in browser
storage, run metadata, launcher logs, or API responses.

The start action asks for confirmation and launches the repository's reviewed
`run-new-live.sh` workflow. That is a paid production workflow and includes
setup, preflight, autonomous execution, finalization, and validation. Launcher
logs are written under ignored `.tacorank/dashboard-launches/`.

## Validate

```bash
npm run lint
npm run build
```
