# Person 3 integration handoff

This implementation provides the coding-worker, Git-lineage, deterministic safety-gate, execution, telemetry, artifact, and prediction-output boundaries defined by `03-person-3-brian-coding-execution-safety.md`.

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

For a data-independent production bring-up, run `tacorank setup-trae`, then `tacorank trae-preflight --config .tacorank/trae/trae-deployment.json --local-only`. After exporting a rotated `DEEPSEEK_API_KEY`, omit `--local-only` to authenticate and verify model access. `tacorank trae-run-example` consumes `examples/trae/experiment-spec.json`, performs one real coding action and Gate A, and stops before ML execution. This is the reviewed Person 3 boundary, not a fake-adapter smoke path.

Production coding is fail-closed unless `TraeConfig` names the reviewed source revision, a hashed VCS `direct_url.json`, a hashed console executable, a digest-pinned image, and an absolute non-symlink Docker CLI. Trae runs from an explicit non-candidate trusted runtime root whose complete installed `trae_agent` package tree—including `dist` assets and attach-path source files—matches a frozen manifest; it never uses the experiment worktree as the host process cwd. Before every external action the adapter rejects any `.env` in that runtime tree or its ancestor search path. `python-dotenv==1.2.2` and its installed metadata are verified, and `PYTHON_DOTENV_DISABLED=1` is forced. Each version/run invocation also receives fresh controller-owned `0700` HOME, temp, XDG config/cache, and Docker-config directories; code-loader, Git, Docker-daemon, and shell startup environment overrides are rejected.

The adapter creates the container itself with no network, a read-only root, dropped capabilities, no-new-privileges, hard RAM/CPU/PID bounds, a single read-write worktree bind, bounded `/tmp`, a read-only `/agent_tools` bind containing only the manifest-verified packaged tools, and no provider credentials. Preflight executes the mounted edit tool inside the same hardened boundary before any ledger event. The adapter passes only the resulting container ID to Trae and verifies removal after every success or failure. `trusted_test_mode=True` is an explicit fake-process test seam and is never a production isolation claim. The hashed YAML is parsed and must retain exactly the reviewed edit/task-done tool set, disabled MCP, and matching provider/model/step limits. Upstream Trae's unrestricted Bash tool is disabled in production: this worker is edit-only and Person 2 routes lightweight checks through the reviewed symbolic execution registry. Bash can be re-enabled only after a separate container-side argv capability wrapper is implemented and attested.

For the pinned `e839e559...` source specifically, `--working-dir` and `--docker-container-id` are intentionally passed together. That revision stores the absolute host worktree as `project_path` for `--must-patch`, passes it to `DockerToolExecutor.host_workspace_dir`, and maps tool paths to the controller-mounted `/workspace`. Newer README guidance is not treated as authority for this pinned code. Docker rejects `docker cp` whenever `ReadonlyRootfs` is set, including copies targeting a tmpfs mount. Trae setup therefore applies a deterministic, manifest-attested compatibility patch to the pinned runtime: when both packaged executables already exist in `/agent_tools`, Trae reuses them instead of issuing its redundant copy. TacoRank mounts only the verified `dist` directory there as read-only, and the unprivileged candidate user cannot modify it.

The pinned Trae OpenAI Responses client also requires a deterministic DeepSeek compatibility patch. The worker dependency file pins the reviewed OpenAI SDK instead of accepting Trae's mutable lower bound. Setup sends `reasoning={"effort": "high"}` for `deepseek-v4-flash` and preserves each returned reasoning item across function-call turns after removing fields that DeepSeek documents as unsupported inputs. The patched client is included in the runtime manifest, and the coding worker refuses a DeepSeek deployment when the attested patch is absent.

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

## Live CPU acceptance evidence

The final acceptance deployment was generated from tracked source commit `5f6373c` as run `person3_cpu_acceptance_010`. Its credentialed preflight passed without creating a ledger and reproduced the official FM baseline primary score `0.601468756352959`. The deployment bound Docker image `sha256:ce35c21d01f3fde056159f0fd2211d356679afddfd44d9f62808c45fa75d598b`, data manifest `0ca4fbf1fc9f1948e752bb7a062d8935dcff57a99e0c9a81c962dc92e7befa97`, the reviewed contract, legal views, and the generated label-free submission rows.

The live production iteration used `deepseek-v4-flash` with high reasoning and no fake adapter:

| Evidence | Observed result |
| --- | --- |
| Trae coding | 178,812 provider-measured input tokens, 27,487 output tokens, 216,322 ms, zero run-recorded manual interventions |
| Trajectory and diff | `775c7a0c80b17a26f0a0a077a73f4ca66e0b42da5fce4b1d76e9bc09551efc16` and `9c3b5918b9ab7b6f913b4e36d1aea7e298ed97afde7c8198f9ba352a7cb72b7b` |
| Gate A | Accepted commit `7e76b05ea59341f7e0c44b61c2401ae7be9313c6`; receipt `485a7aca2d90d11f305de3e5a3e33e83575b96bd7e1e721667a67d4b49a69a7a` |
| Smoke execution | Exit 0; Gate B accepted all 11 checks; prediction SHA-256 `e3fc223d75656d670aa4c604dc1e86cde784d1131b3c6c9ccada7d3c2f400398` |
| Proxy execution | Exit 0; Gate B accepted all 11 checks; GAUC `0.6121441544192503`, nDCG@5 `0.5086155968924914`, primary `0.5603798756558709` |
| Controller decision | `PROXY_FAILED`; correctly pruned because the candidate underperformed the frozen baseline |

Because the honest proxy decision prevented full promotion, full-fidelity execution was validated separately against the same Gate-A-sealed commit. This diagnostic used the canonical production runner, execution seal, Gate B, protected evaluator, and submission checker, but did not append a false promotion or full-evaluation event to the research ledger.

| Full-fidelity check | Observed result |
| --- | --- |
| Repetitions | Attempts 3 and 4, both seed 33, CPU only |
| Determinism | Identical prediction SHA-256 `c5a47927fbb8f3fe1ce706c54c99b78d730ac2756777814389108a36bb101cdb` and ordered prediction SHA-256 `0c5cc58785ae97c22de941f02f6a70c15c82f522ffebab3ac06515a4501a5d14` |
| Gate B | Accepted twice; artifact identity, header, types, row count/order, duplicate preservation, finite/diverse scores, producer seal, and protected-data checks all passed |
| Official full metrics | GAUC `0.6262603394573768`, nDCG@5 `0.5177471235029993`, primary `0.572003731480188` |
| Submission compatibility | Protected checker passed all 124,909 rows |
| Limits | 2 Docker CPUs, 4096 MB request limit, 600 s wall limit, 128 processes, 256 open files, network disabled, zero GPUs, 256 MB runtime tmpfs, 2 GiB hard output quota |
| Telemetry | 3 and 4 samples; peak RSS 97 MB; peak observed CPU 101.51%; no GPU samples |

Run artifacts, the official dataset, credentials, and generated deployment files remain ignored and are not Git deliverables. Reviewers can reproduce the stage from a clean checkout with `setup-live`, `preflight`, and `run`. The evidence above validates Person 3's coding and CPU execution boundary; it is not evidence that the complete multi-experiment search converged, and it does not remove the documented `resume` and `finalize` limitations.
