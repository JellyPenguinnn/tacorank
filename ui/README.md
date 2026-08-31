# TacoRank Run Dashboard

Local repository-backed dashboard for starting and inspecting TacoRank runs.
It lists every ledger under `runs/`, opens the latest run by default, refreshes
every five seconds, and shows each iteration's plan, memory/context, patch,
Gate A, execution, Gate B, evaluation, recovery, and raw ledger evidence. The
current experiment and iteration cards show whether the proposal starts from
the baseline or continues from a named parent experiment. The
current-run panel updates elapsed stage/iteration time and last-event age every
second from ledger timestamp anchors, and shows the active attempt, fidelity,
timeout, deadline, and loop step. A dashboard launch appears immediately with
its reserved run ID while setup and non-mutating preflight run; the placeholder
is replaced by authoritative ledger data when `events.jsonl` is created. If a
controller disappears after a stage deadline, the dashboard marks the run as
interrupted and freezes its runtime at the last durable event.

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

On macOS, the launcher discovers Docker Desktop and adds its CLI directory to
the child `PATH` so `docker-credential-desktop` remains available during image
builds. Docker Desktop itself must be running before a production launch.

The start action asks for confirmation and launches the repository's reviewed
platform-specific workflow: `run-new-live.ps1` through non-interactive Windows
PowerShell on Windows, or `run-new-live.sh` on macOS/Linux. Both receive the
same dashboard-reserved run ID and perform the paid production workflow:
setup, preflight, autonomous execution, finalization, and validation. Launcher
logs are written under ignored `.tacorank/dashboard-launches/`.

The start dialog defaults to the 50-slot objective-then-temporal campaign and
also exposes the standard autonomous search as an explicit alternative.

## Validate

```bash
npm run lint
npm run build
```
