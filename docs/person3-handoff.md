# Person 3 integration handoff

This branch implements the coding-worker, Git-lineage, deterministic safety-gate, execution, telemetry, artifact, and prediction-output boundaries defined by `03-person-3-brian-coding-execution-safety.md`.

## Ownership and authority

Person 3 owns code under:

- `src/tacorank/coding/`
- `src/tacorank/git/`
- `src/tacorank/safety/`
- `src/tacorank/execution/`
- the corresponding test directories

Person 2 remains the sole owner of `src/tacorank/schemas.py`, orchestration, event persistence, budgets, and final routing. Person 3 does not choose experiments, classify recovery, compute official metrics, update best refs, or append events.

All production adapters resolve Person 2's shared models at their integration boundary. The branch is tested against the canonical models now present in `src/tacorank/schemas.py`; if a required model is unavailable, it fails before invoking Trae or starting a score-bearing process. `CanonicalArtifactStoreAdapter` connects Person 2's `ArtifactStore` to the execution port without duplicating artifact ownership. Test-only factories exercise the Person 3 mechanisms without introducing replacement production records.

For Person 2 orchestration tests, the public packages also export `FakeCodingWorker`, `FakePatchGate`, `FakeExecutionRunner`, and `FakeOutputGate`. They return caller-supplied canonical shared models and record calls; they never define replacement schemas.

## Required upstream inputs

Before live execution, the controller must supply:

1. a frozen contract and protected-path manifest with verified SHA-256 digests;
2. a data manifest and legal data views for each fidelity;
3. controller-assigned `attempt` and `experiment_spec_event_id` values for every coding action;
4. an allowlisted symbolic command configuration;
5. a `HealthObserver` implementation from Person 4.

The Trae executable/version and provider/model identity are deployment configuration. Credentials may enter only through the approved child-process environment and must never be persisted. Install the separately isolated Python 3.12+ worker from `requirements-trae.txt`; it pins the reviewed upstream source commit. Copy `config/trae-agent.yaml.example` outside Git, choose the controller-approved provider/model, hash the final credential-free file, and pass that exact path/hash through `TraeConfig`.

Production coding is fail-closed unless `TraeConfig` names the reviewed source revision, a hashed VCS `direct_url.json`, a hashed console executable, a digest-pinned image, and an absolute non-symlink Docker CLI. Trae runs from an explicit non-candidate trusted runtime root whose complete installed `trae_agent` package tree—including `dist` assets and attach-path source files—matches a frozen manifest; it never uses the experiment worktree as the host process cwd. Before every external action the adapter rejects any `.env` in that runtime tree or its ancestor search path. `python-dotenv==1.2.2` and its installed metadata are verified, and `PYTHON_DOTENV_DISABLED=1` is forced. Each version/run invocation also receives fresh controller-owned `0700` HOME, temp, XDG config/cache, and Docker-config directories; code-loader, Git, Docker-daemon, and shell startup environment overrides are rejected.

The adapter creates the container itself with no network, a read-only root, dropped capabilities, no-new-privileges, hard RAM/CPU/PID bounds, a single read-write worktree bind, bounded `/tmp`, a read-only `/agent_tools` bind containing only the manifest-verified packaged tools, and no provider credentials. Preflight executes the mounted edit tool inside the same hardened boundary before any ledger event. The adapter passes only the resulting container ID to Trae and verifies removal after every success or failure. `trusted_test_mode=True` is an explicit fake-process test seam and is never a production isolation claim. The hashed YAML is parsed and must retain exactly the reviewed edit/task-done tool set, disabled MCP, and matching provider/model/step limits. Upstream Trae's unrestricted Bash tool is disabled in production: this worker is edit-only and Person 2 routes lightweight checks through the reviewed symbolic execution registry. Bash can be re-enabled only after a separate container-side argv capability wrapper is implemented and attested.

For the pinned `e839e559...` source specifically, `--working-dir` and `--docker-container-id` are intentionally passed together. That revision stores the absolute host worktree as `project_path` for `--must-patch`, passes it to `DockerToolExecutor.host_workspace_dir`, and maps tool paths to the controller-mounted `/workspace`. Newer README guidance is not treated as authority for this pinned code. Docker rejects `docker cp` whenever `ReadonlyRootfs` is set, including copies targeting a tmpfs mount. `setup-live` therefore applies a deterministic, manifest-attested compatibility patch to the pinned Trae runtime: when both packaged executables already exist in `/agent_tools`, Trae reuses them instead of issuing its redundant copy. TacoRank mounts only the verified `dist` directory there as read-only, and the unprivileged candidate user cannot modify it.

If a frozen commit contains gitlinks, construct `WorktreeManager` with the exact `required_submodules=(...)` allowlist. Creation uses only the already-present local `.git/modules` object store with `--no-fetch`, ignores the recorded URL for transport, verifies each exact gitlink commit and clean checkout, and fails closed when local objects are unavailable. Candidate patches cannot change submodule refs.

Coding and execution share `WorktreeManager.acquire_lease(record, timeout_seconds=...)`. Production holds this exclusive OS-backed lease across Trae verification/edit/commit and across execution preverification/process cleanup/postverification, preventing transient modify-and-restore races. Acquisition is bounded; lock files are private and no-follow, while kernel `flock` ownership is released automatically on process death.

## Sealed execution sequence

```text
approved ExperimentSpec
  -> disposable experiment/<run_id>/<experiment_id> worktree
  -> bounded Trae invocation and exact diff/trajectory capture
  -> deterministic Gate A checks
  -> commit/diff/hash-bound verification receipt
  -> allowlisted command in a sanitized process group or container
  -> telemetry samples passed to Person 4
  -> hash-addressed run artifacts and typed RunResult
  -> deterministic Gate B prediction checks
  -> accepted OutputCheckResult returned to Person 2
```

No execution may consume a raw LLM command. A repaired commit invalidates the earlier receipt and must pass Gate A again.

## Symbolic commands

The registry recognizes these contract-owned IDs:

- `baseline_full`
- `candidate_smoke`
- `candidate_proxy`
- `candidate_full`
- `candidate_final_infer`
- `submission_check`
- `clean_reproduce`

Registry entries resolve immutable argv, working directory, environment allowlist, expected artifacts, and resource profile. Commands execute with `shell=False`; network is disabled unless both the frozen entry and `RunRequest` allow it.

The controller-owned `tacorank.execution.solution_cli` is the research-neutral fail-closed CLI; it is deliberately outside the Trae-editable `solution/` tree. A reviewed command registry must supply canonical contract/input/artifact roots and callable entrypoints through `PipelineCommandInputs`. Person 1 supplies the approved research specification, Person 2 freezes and routes contract/data identities and reviewed entrypoints, and Person 5 receives predictions only after Person 3's structural gate; the adapter does not guess missing values or compute metrics.

`submission_check` consumes a previously verified prediction through a controller-provided `SubmissionArtifactResolver`; the runner mounts that exact regular file read-only and never assumes a prediction exists in a fresh attempt. `candidate_final_infer` and `submission_check` use the shared `full` RunRequest fidelity while their distinct command IDs retain final-inference semantics. Gate B additionally requires an `ExecutionSealExpectation` built from controller-owned run, command, data-manifest, commit, and Gate A receipt identities. Missing or mismatched seal evidence is rejected before evaluation.

Production Docker execution requires an attested credential-free image environment and a production-capable hard output-quota verifier. The included `DedicatedFilesystemQuotaVerifier` accepts only an exact dedicated output mount whose total filesystem capacity is within the configured byte cap. GPU commands currently fail closed because the Docker backend cannot prove a hard per-container GPU-memory ceiling; enable them only after integrating an enforcement backend that can provide that guarantee. `TrustedLocalProcessSandbox(..., allow_unsafe_for_tests=True)` is strictly a test adapter, not a production fallback.

## Artifact contract

Attempt-local evidence is stored below:

```text
runs/<run_id>/artifacts/<experiment_id>/attempt_<nnn>/
```

Every returned artifact reference contains a normalized relative path, lowercase SHA-256, byte size, and content type. Full logs and telemetry stay in artifacts; typed handoffs carry only bounded summaries.

## Development checks

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pytest tests/coding tests/git tests/safety tests/execution tests/failure_injection tests/integration
```

The failure-injection subset can be run independently with:

```bash
python -m pytest tests/failure_injection
```

The core TacoRank package supports Python 3.9+. Trae Agent is an external process with its own Python 3.12+ environment:

```bash
python3.12 -m pip install -r requirements-trae.txt
```

Live Trae, container, GPU, and full-data validation are separate deployment checks and must not be inferred from mocked tests.
