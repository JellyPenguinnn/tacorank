# TacoRank Run Monitor

Read-only web dashboard for TacoRank run evidence. It starts with clearly
labelled preview data and can monitor a real run without uploading its files.

## Run locally

```bash
pnpm install
pnpm dev
```

Open the printed local URL, select **Connect run folder**, and choose a
`runs/<run_id>` directory. The page reads `state.json` and `events.jsonl` in
the browser and refreshes them every five seconds. Browsers without directory
access can import those two files as a point-in-time snapshot.

## Validate

```bash
pnpm lint
pnpm build
```
