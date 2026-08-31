import 'server-only';

import { existsSync, promises as fs } from 'node:fs';
import path from 'node:path';

export type JsonRecord = Record<string, unknown>;

export type LedgerEvent = {
  event_id?: string;
  seq?: number;
  timestamp?: string;
  run_id?: string;
  event_type?: string;
  event_hash?: string;
  payload?: JsonRecord;
  artifact_refs?: JsonRecord[];
  resource_delta?: JsonRecord;
};

export type RunSummary = {
  run_id: string;
  source: 'ledger' | 'launch';
  status: string;
  phase: string;
  is_live: boolean;
  can_stop: boolean;
  stop_requested_at: string | null;
  launch_error: string | null;
  started_at: string | null;
  updated_at: string | null;
  experiments_proposed: number;
  best_experiment_id: string | null;
  best_primary_score: number | null;
  best_primary_fidelity: string | null;
  baseline_primary_score: number | null;
  stop_reason_code: string | null;
  final_experiment_id: string | null;
  event_count: number;
  current_experiment_id: string | null;
  current_attempt: number | null;
  current_fidelity: string | null;
  stage_started_at: string | null;
  configured_timeout_seconds: number | null;
  estimated_deadline: string | null;
  last_event_id: string | null;
  last_event_type: string | null;
  last_event_at: string | null;
};

export type LaunchRecord = {
  schema_version: 'tacorank.dashboard-launch.v1';
  launch_id: string;
  run_id: string;
  started_at: string;
  pid: number | null;
  stop_requested_at: string | null;
};

const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LAUNCH_ID = /^[0-9]{17}$/;
const MAX_ARTIFACT_BYTES = 512_000;
const MAX_LAUNCH_RECORD_BYTES = 16_384;
const MAX_LAUNCH_LOG_TAIL_BYTES = 65_536;
const MAX_LAUNCH_LOG_HEADER_BYTES = 4_096;
const LAUNCH_START_GRACE_MS = 10_000;

export function repositoryRoot(): string {
  const configured = process.env.TACORANK_REPOSITORY_ROOT;
  if (configured) return path.resolve(configured);
  const workingDirectory = process.cwd();
  return existsSync(path.join(workingDirectory, 'src', 'tacorank'))
    ? workingDirectory
    : path.resolve(workingDirectory, '..');
}

function record(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {};
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value.length ? value : null;
}

function number(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function nested(payload: JsonRecord, name: string): JsonRecord {
  return record(payload[name]);
}

export function assertRunId(runId: string): void {
  if (!RUN_ID.test(runId)) throw new Error('Invalid run ID.');
}

function launchDirectory(): string {
  return path.join(repositoryRoot(), '.tacorank', 'dashboard-launches');
}

function validLaunchRecord(value: unknown): LaunchRecord | null {
  const candidate = record(value);
  const launchId = text(candidate.launch_id);
  const runId = text(candidate.run_id);
  const startedAt = text(candidate.started_at);
  const stopRequestedAt = candidate.stop_requested_at == null ? null : text(candidate.stop_requested_at);
  const pid = candidate.pid;
  if (
    candidate.schema_version !== 'tacorank.dashboard-launch.v1'
    || !launchId || !LAUNCH_ID.test(launchId)
    || !runId || !RUN_ID.test(runId)
    || !startedAt || !Number.isFinite(new Date(startedAt).getTime())
    || (candidate.stop_requested_at != null && (!stopRequestedAt || !Number.isFinite(new Date(stopRequestedAt).getTime())))
    || !(pid === null || (typeof pid === 'number' && Number.isInteger(pid) && pid > 0))
  ) return null;
  return {
    schema_version: 'tacorank.dashboard-launch.v1',
    launch_id: launchId,
    run_id: runId,
    started_at: startedAt,
    pid: pid as number | null,
    stop_requested_at: stopRequestedAt,
  };
}

export async function writeLaunchRecord(value: LaunchRecord): Promise<void> {
  const parsed = validLaunchRecord(value);
  if (!parsed) throw new Error('Invalid dashboard launch record.');
  const directory = launchDirectory();
  await fs.mkdir(directory, { recursive: true });
  const destination = path.join(directory, `${parsed.launch_id}.json`);
  const temporary = path.join(directory, `.${parsed.launch_id}.${process.pid}.tmp`);
  await fs.writeFile(temporary, `${JSON.stringify(parsed)}\n`, { encoding: 'utf8', mode: 0o600 });
  await fs.rename(temporary, destination);
}

async function readLaunchRecords(): Promise<LaunchRecord[]> {
  const directory = launchDirectory();
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  const records = await Promise.all(entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json') && LAUNCH_ID.test(entry.name.slice(0, -5)))
    .map(async (entry) => {
      try {
        const location = path.join(directory, entry.name);
        const stat = await fs.stat(location);
        if (!stat.isFile() || stat.size > MAX_LAUNCH_RECORD_BYTES) return null;
        return validLaunchRecord(JSON.parse(await fs.readFile(location, 'utf8')));
      } catch {
        return null;
      }
    }));
  return records.filter((item): item is LaunchRecord => item !== null);
}

function launchIdTimestamp(launchId: string): string | null {
  const match = /^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{3})$/.exec(launchId);
  if (!match) return null;
  const [, year, month, day, hour, minute, second, millisecond] = match;
  const value = `${year}-${month}-${day}T${hour}:${minute}:${second}.${millisecond}Z`;
  return Number.isFinite(new Date(value).getTime()) ? value : null;
}

async function activeLauncherPid(): Promise<number | null> {
  try {
    const value = (await fs.readFile(path.join(repositoryRoot(), '.tacorank', 'live-run.lock', 'pid'), 'utf8')).trim();
    if (!/^[1-9][0-9]*$/.test(value)) return null;
    const pid = Number(value);
    return Number.isSafeInteger(pid) && processIsAlive(pid) ? pid : null;
  } catch {
    return null;
  }
}

async function legacyActiveLaunch(existing: LaunchRecord[]): Promise<LaunchRecord | null> {
  const pid = await activeLauncherPid();
  if (pid === null || existing.some((item) => item.pid === pid)) return null;
  const directory = launchDirectory();
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  for (const entry of entries
    .filter((item) => item.isFile() && item.name.endsWith('.log') && LAUNCH_ID.test(item.name.slice(0, -4)))
    .sort((left, right) => right.name.localeCompare(left.name))) {
    try {
      const launchId = entry.name.slice(0, -4);
      const handle = await fs.open(path.join(directory, entry.name), 'r');
      try {
        const stat = await handle.stat();
        const size = Math.min(stat.size, MAX_LAUNCH_LOG_HEADER_BYTES);
        const buffer = Buffer.alloc(size);
        await handle.read(buffer, 0, size, 0);
        const match = /^Starting new TacoRank live run: ([A-Za-z0-9][A-Za-z0-9._-]{0,127})$/m.exec(buffer.toString('utf8'));
        const startedAt = launchIdTimestamp(launchId);
        if (!match || !startedAt) continue;
        return {
          schema_version: 'tacorank.dashboard-launch.v1',
          launch_id: launchId,
          run_id: match[1],
          started_at: startedAt,
          pid,
          stop_requested_at: null,
        };
      } finally {
        await handle.close();
      }
    } catch {
      // Ignore malformed or unreadable historical launcher logs.
    }
  }
  return null;
}

async function dashboardLaunches(): Promise<LaunchRecord[]> {
  const records = await readLaunchRecords();
  const legacy = await legacyActiveLaunch(records);
  return legacy ? [...records, legacy] : records;
}

function processIsAlive(pid: number | null): boolean {
  if (pid === null) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !(error && typeof error === 'object' && 'code' in error && error.code === 'ESRCH');
  }
}

function launchIsAlive(item: LaunchRecord, observedAt = Date.now()): boolean {
  return processIsAlive(item.pid)
    || (item.pid === null && observedAt - new Date(item.started_at).getTime() < LAUNCH_START_GRACE_MS);
}

export async function hasActiveDashboardLaunch(): Promise<boolean> {
  return (await dashboardLaunches()).some((item) => launchIsAlive(item));
}

export class DashboardStopError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'DashboardStopError';
    this.status = status;
  }
}

export async function requestDashboardStop(runId: string): Promise<{ run_id: string; status: 'stopping'; requested_at: string }> {
  assertRunId(runId);
  const launch = (await dashboardLaunches()).find((item) => item.run_id === runId);
  if (!launch) throw new DashboardStopError('This run was not started by the dashboard and cannot be stopped here.', 409);
  const lockedPid = await activeLauncherPid();
  if (launch.pid === null || lockedPid === null || launch.pid !== lockedPid || !processIsAlive(launch.pid)) {
    throw new DashboardStopError('The controller for this run is no longer active.', 409);
  }
  if (launch.stop_requested_at !== null) {
    return { run_id: runId, status: 'stopping', requested_at: launch.stop_requested_at };
  }

  const requestedAt = new Date().toISOString();
  const requested = { ...launch, stop_requested_at: requestedAt };
  await writeLaunchRecord(requested);
  try {
    if (process.platform === 'win32') process.kill(launch.pid, 'SIGINT');
    else process.kill(-launch.pid, 'SIGINT');
  } catch (error) {
    await writeLaunchRecord(launch).catch(() => undefined);
    const message = error instanceof Error ? error.message : 'Could not signal the run controller.';
    throw new DashboardStopError(`Could not stop ${runId}: ${message}`, 500);
  }
  return { run_id: runId, status: 'stopping', requested_at: requestedAt };
}

async function launchFailure(item: LaunchRecord): Promise<string | null> {
  const location = path.join(launchDirectory(), `${item.launch_id}.log`);
  try {
    const handle = await fs.open(location, 'r');
    try {
      const stat = await handle.stat();
      const size = Math.min(stat.size, MAX_LAUNCH_LOG_TAIL_BYTES);
      const buffer = Buffer.alloc(size);
      await handle.read(buffer, 0, size, stat.size - size);
      const lines = buffer.toString('utf8').split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      return lines.reverse().find((line) => line.startsWith('error:')) ?? null;
    } finally {
      await handle.close();
    }
  } catch {
    return null;
  }
}

async function summarizeLaunch(item: LaunchRecord, lockedPid: number | null): Promise<RunSummary> {
  const isLive = launchIsAlive(item);
  const stopping = isLive && item.stop_requested_at !== null;
  const stopped = !isLive && item.stop_requested_at !== null;
  return {
    run_id: item.run_id,
    source: 'launch',
    status: stopping ? 'stopping' : stopped ? 'interrupted' : isLive ? 'initializing' : 'failed',
    phase: stopping ? 'setup_stopping' : stopped ? 'setup_stopped' : isLive ? 'setup' : 'setup_failed',
    is_live: isLive,
    can_stop: isLive && !stopping && item.pid !== null && item.pid === lockedPid,
    stop_requested_at: item.stop_requested_at,
    launch_error: isLive || stopped ? null : await launchFailure(item),
    started_at: item.started_at,
    updated_at: item.started_at,
    experiments_proposed: 0,
    best_experiment_id: null,
    best_primary_score: null,
    best_primary_fidelity: null,
    baseline_primary_score: null,
    stop_reason_code: null,
    final_experiment_id: null,
    event_count: 0,
    current_experiment_id: null,
    current_attempt: null,
    current_fidelity: null,
    stage_started_at: item.started_at,
    configured_timeout_seconds: null,
    estimated_deadline: null,
    last_event_id: null,
    last_event_type: null,
    last_event_at: null,
  };
}

export async function readEvents(runId: string): Promise<LedgerEvent[]> {
  assertRunId(runId);
  const ledger = path.join(repositoryRoot(), 'runs', runId, 'events.jsonl');
  const contents = await fs.readFile(ledger, 'utf8');
  return contents.split(/\r?\n/).filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line) as LedgerEvent;
    } catch {
      throw new Error(`Invalid JSON in ${runId}/events.jsonl at line ${index + 1}.`);
    }
  });
}

function experimentId(event: LedgerEvent): string | null {
  const payload = record(event.payload);
  for (const candidate of [payload.spec, payload.candidate, payload.result, payload.request, payload.decision]) {
    const id = text(record(candidate).experiment_id);
    if (id) return id;
  }
  return text(payload.experiment_id);
}

function metricScore(payload: JsonRecord): number | null {
  const result = nested(payload, 'result');
  const trust = record(result.trust);
  const metricSet = record(result.metric_set ?? payload.metric_set);
  return number(trust.seed_mean ?? metricSet.primary_score ?? payload.primary_score);
}

export function summarizeRun(runId: string, events: LedgerEvent[]): RunSummary {
  let status = 'initializing';
  let phase = 'not_started';
  let proposed = 0;
  let bestId: string | null = null;
  let bestScore: number | null = null;
  let bestFidelity: string | null = null;
  let baselineScore: number | null = null;
  let stopReason: string | null = null;
  let finalId: string | null = null;
  let activeExperiment: string | null = null;
  let activeAttempt: number | null = null;
  let activeFidelity: string | null = null;
  let stageStartedAt: string | null = null;
  let coderTimeout: number | null = null;
  let executionTimeout: number | null = null;
  for (const event of events) {
    const payload = record(event.payload);
    const previousPhase = phase;
    switch (event.event_type) {
      case 'run.started': phase = 'contract_verification'; break;
      case 'contract.verified': phase = 'baseline'; break;
      case 'baseline.verified':
        status = 'ready'; phase = 'planning'; baselineScore = metricScore(payload);
        bestId = 'baseline'; bestScore = baselineScore; bestFidelity = 'full';
        break;
      case 'context.created': {
        const context = nested(payload, 'context');
        phase = `${text(context.role) ?? 'planner'}_context`;
        if (context.role === 'coder') coderTimeout = number(context.wall_time_limit_seconds);
        break;
      }
      case 'experiment.proposed': {
        const spec = nested(payload, 'spec');
        status = 'running'; phase = 'coding'; proposed += 1;
        activeExperiment = text(spec.experiment_id); activeAttempt = 1; activeFidelity = null;
        break;
      }
      case 'patch.created': {
        const candidate = nested(payload, 'candidate');
        phase = 'patch_gate'; activeAttempt = number(candidate.attempt); break;
      }
      case 'patch.checked': phase = nested(payload, 'result').accepted === false ? 'recovery' : 'execution'; break;
      case 'execution.started': {
        const request = nested(payload, 'request');
        phase = 'running'; activeAttempt = number(request.attempt);
        activeFidelity = text(request.fidelity); executionTimeout = number(request.timeout_seconds); break;
      }
      case 'execution.finished': phase = text(nested(payload, 'result').outcome) === 'success' ? 'output_gate' : 'recovery'; break;
      case 'adapter.failed': phase = 'recovery'; break;
      case 'recovery.decided': phase = 'recovery'; break;
      case 'output.checked': phase = nested(payload, 'result').accepted === false ? 'recovery' : 'evaluation'; break;
      case 'evaluation.completed': {
        phase = 'decision';
        break;
      }
      case 'experiment.decided': phase = 'planning'; break;
      case 'best.updated':
        bestId = text(payload.experiment_id);
        bestScore = number(payload.primary_score);
        bestFidelity = 'full';
        break;
      case 'run.stopped':
        status = 'stopped'; phase = 'stopped'; stopReason = text(payload.reason_code);
        activeExperiment = null; activeAttempt = null; activeFidelity = null; break;
      case 'final.selected': status = 'finalizing'; phase = 'submission'; finalId = text(payload.experiment_id); break;
      case 'submission.checked':
        status = payload.accepted === true ? 'finalized' : 'failed'; phase = status; break;
    }
    if (phase !== previousPhase) stageStartedAt = event.timestamp ?? null;
  }
  const timeout = phase === 'coder_context' ? coderTimeout : phase === 'running' ? executionTimeout : null;
  const deadline = timeout !== null && stageStartedAt
    ? new Date(new Date(stageStartedAt).getTime() + timeout * 1000).toISOString()
    : null;
  const lastEvent = events.at(-1);
  return {
    run_id: runId,
    source: 'ledger',
    status,
    phase,
    is_live: ['initializing', 'ready', 'running', 'finalizing'].includes(status),
    can_stop: false,
    stop_requested_at: null,
    launch_error: null,
    started_at: events[0]?.timestamp ?? null,
    updated_at: events.at(-1)?.timestamp ?? null,
    experiments_proposed: proposed,
    best_experiment_id: bestId,
    best_primary_score: bestScore,
    best_primary_fidelity: bestFidelity,
    baseline_primary_score: baselineScore,
    stop_reason_code: stopReason,
    final_experiment_id: finalId,
    event_count: events.length,
    current_experiment_id: activeExperiment,
    current_attempt: activeAttempt,
    current_fidelity: activeFidelity,
    stage_started_at: stageStartedAt,
    configured_timeout_seconds: timeout,
    estimated_deadline: deadline,
    last_event_id: lastEvent?.event_id ?? null,
    last_event_type: lastEvent?.event_type ?? null,
    last_event_at: lastEvent?.timestamp ?? null,
  };
}

function withObservedRuntime(summary: RunSummary, launch: LaunchRecord | undefined, lockedPid: number | null): RunSummary {
  if (!summary.is_live) return summary;
  if (launch) {
    // Once a ledger exists, the controller-owned lock is the authoritative
    // liveness signal. A detached launcher PID can remain observable briefly
    // after exit (or be reused), which otherwise leaves the dashboard clock
    // running forever against a dead controller.
    const alive = launchIsAlive(launch)
      && launch.pid !== null
      && lockedPid !== null
      && launch.pid === lockedPid;
    if (launch.stop_requested_at !== null) {
      return {
        ...summary,
        status: alive ? 'stopping' : 'interrupted',
        is_live: alive,
        can_stop: false,
        stop_requested_at: launch.stop_requested_at,
      };
    }
    if (!alive) return { ...summary, status: 'interrupted', is_live: false, can_stop: false };
    return {
      ...summary,
      can_stop: launch.pid !== null && launch.pid === lockedPid,
    };
  }
  if (!summary.estimated_deadline) return summary;
  const deadline = new Date(summary.estimated_deadline).getTime();
  if (!Number.isFinite(deadline) || Date.now() <= deadline) return summary;
  return {
    ...summary,
    status: 'interrupted',
    is_live: false,
  };
}

function durationSeconds(start?: string, finish?: string): number | null {
  if (!start || !finish) return null;
  return Math.max(0, (new Date(finish).getTime() - new Date(start).getTime()) / 1000);
}

function experimentTiming(events: LedgerEvent[]): JsonRecord {
  const proposal = events.find((event) => event.event_type === 'experiment.proposed');
  const terminal = events.find((event) => {
    const payload = record(event.payload);
    if (event.event_type === 'experiment.decided') return text(nested(payload, 'decision').decision) !== 'promote';
    if (event.event_type === 'recovery.decided') return ['abandon', 'rollback'].includes(text(nested(payload, 'decision').action) ?? '');
    return false;
  });
  let codingStart: LedgerEvent | undefined;
  let executionStart: LedgerEvent | undefined;
  let codingSeconds = 0;
  let executionSeconds = 0;
  let recoverySeconds = 0;
  const byId = new Map(events.map((event) => [event.event_id, event]));
  for (const event of events) {
    const payload = record(event.payload);
    if (event.event_type === 'context.created' && nested(payload, 'context').role === 'coder') codingStart = event;
    else if (event.event_type === 'recovery.decided' && nested(payload, 'decision').action === 'trae_repair') codingStart = event;
    else if (event.event_type === 'patch.created' && codingStart) {
      codingSeconds += durationSeconds(codingStart.timestamp, event.timestamp) ?? 0; codingStart = undefined;
    } else if (event.event_type === 'adapter.failed' && nested(payload, 'result').failure_stage === 'coding' && codingStart) {
      codingSeconds += durationSeconds(codingStart.timestamp, event.timestamp) ?? 0; codingStart = undefined;
    }
    if (event.event_type === 'execution.started') executionStart = event;
    else if (event.event_type === 'execution.finished' && executionStart) {
      executionSeconds += durationSeconds(executionStart.timestamp, event.timestamp) ?? 0; executionStart = undefined;
    } else if (event.event_type === 'adapter.failed' && nested(payload, 'result').failure_stage === 'execution' && executionStart) {
      executionSeconds += durationSeconds(executionStart.timestamp, event.timestamp) ?? 0; executionStart = undefined;
    }
    if (event.event_type === 'recovery.decided') {
      const failure = byId.get(text(nested(payload, 'decision').failure_event_id) ?? undefined);
      if (failure) recoverySeconds += durationSeconds(failure.timestamp, event.timestamp) ?? 0;
    }
  }
  return {
    proposed_at: proposal?.timestamp ?? null,
    terminal_at: terminal?.timestamp ?? null,
    terminal_event_id: terminal?.event_id ?? null,
    loop_time_seconds: durationSeconds(proposal?.timestamp, terminal?.timestamp),
    trae_coding_time_seconds: codingSeconds,
    execution_time_seconds: executionSeconds,
    recovery_time_seconds: recoverySeconds,
  };
}

async function safeArtifactText(reference: unknown): Promise<string | null> {
  const relative = text(record(reference).path);
  if (!relative) return null;
  const root = repositoryRoot();
  const resolved = path.resolve(root, relative);
  if (!resolved.startsWith(`${root}${path.sep}`)) return null;
  try {
    const stat = await fs.stat(resolved);
    if (!stat.isFile() || stat.size > MAX_ARTIFACT_BYTES) return null;
    return await fs.readFile(resolved, 'utf8');
  } catch {
    return null;
  }
}

export async function listRuns(): Promise<RunSummary[]> {
  const launches = await dashboardLaunches();
  const lockedPid = await activeLauncherPid();
  const launchByRunId = new Map(launches.map((item) => [item.run_id, item]));
  const directory = path.join(repositoryRoot(), 'runs');
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  const runs = await Promise.all(entries
    .filter((entry) => entry.isDirectory() && RUN_ID.test(entry.name))
    .map(async (entry) => {
      try { return withObservedRuntime(summarizeRun(entry.name, await readEvents(entry.name)), launchByRunId.get(entry.name), lockedPid); }
      catch { return null; }
    }));
  const ledgerRuns = runs.filter((run): run is RunSummary => run !== null);
  const ledgerRunIds = new Set(ledgerRuns.map((run) => run.run_id));
  const pendingLaunches = await Promise.all(launches
    .filter((item) => !ledgerRunIds.has(item.run_id))
    .map((item) => summarizeLaunch(item, lockedPid)));
  return [...ledgerRuns, ...pendingLaunches]
    .sort((a, b) => (b.updated_at ?? '').localeCompare(a.updated_at ?? ''));
}

export async function runDetail(runId: string): Promise<JsonRecord> {
  assertRunId(runId);
  const ledger = path.join(repositoryRoot(), 'runs', runId, 'events.jsonl');
  if (!existsSync(ledger)) {
    const launches = await dashboardLaunches();
    const launch = launches.find((item) => item.run_id === runId);
    if (!launch) throw new Error('Run not found.');
    return {
      summary: await summarizeLaunch(launch, await activeLauncherPid()),
      events: [],
      iterations: [],
      memory: { planner_contexts: [], lessons: [] },
    };
  }
  const events = await readEvents(runId);
  const launch = (await dashboardLaunches()).find((item) => item.run_id === runId);
  const summary = withObservedRuntime(summarizeRun(runId, events), launch, await activeLauncherPid());
  const iterations = new Map<string, JsonRecord>();
  const plannerContexts: JsonRecord[] = [];
  const lessons: JsonRecord[] = [];

  for (const event of events) {
    const payload = record(event.payload);
    if (event.event_type === 'context.created') {
      const context = nested(payload, 'context');
      if (context.role === 'planner') plannerContexts.push(context);
      const id = text(context.experiment_id);
      if (id) {
        const item = iterations.get(id) ?? { experiment_id: id, events: [], contexts: [] };
        (item.contexts as JsonRecord[]).push(context);
        iterations.set(id, item);
      }
    }
    if (event.event_type === 'lesson.recorded') lessons.push(payload);
    const id = experimentId(event);
    if (!id || id === 'baseline') continue;
    const item = iterations.get(id) ?? { experiment_id: id, events: [], contexts: [] };
    (item.events as LedgerEvent[]).push(event);
    if (event.event_type === 'experiment.proposed') item.plan = payload.spec;
    if (event.event_type === 'patch.created') {
      const candidate = nested(payload, 'candidate');
      item.implementation = {
        ...candidate,
        diff: await safeArtifactText(candidate.diff_artifact),
      };
    }
    if (event.event_type === 'patch.checked') item.gate_a = payload.result;
    if (event.event_type === 'execution.started') item.execution_request = payload.request;
    if (event.event_type === 'execution.finished') item.execution_result = payload.result;
    if (event.event_type === 'output.checked') item.gate_b = payload.result;
    if (event.event_type === 'evaluation.completed') {
      item.evaluation = {
        ...nested(payload, 'result'),
        reported_primary_score: metricScore(payload),
      };
    }
    if (event.event_type === 'experiment.decided') item.decision = payload.decision;
    if (event.event_type === 'adapter.failed') {
      const failures = Array.isArray(item.failures) ? item.failures as JsonRecord[] : [];
      failures.push(nested(payload, 'result'));
      item.failures = failures;
    }
    if (event.event_type === 'recovery.decided') {
      const recoveries = Array.isArray(item.recoveries) ? item.recoveries as JsonRecord[] : [];
      recoveries.push(nested(payload, 'decision'));
      item.recoveries = recoveries;
    }
    iterations.set(id, item);
  }

  for (const item of iterations.values()) {
    const plan = record(item.plan);
    const contextId = text(plan.context_id);
    const contexts = item.contexts as JsonRecord[];
    const plannerContext = plannerContexts.find((context) => text(context.context_id) === contextId);
    if (plannerContext && !contexts.some((context) => text(context.context_id) === contextId)) {
      contexts.unshift(plannerContext);
    }
    const eventIds = new Set((item.events as LedgerEvent[]).map((event) => event.event_id).filter(Boolean));
    item.lessons = lessons.filter((lesson) => {
      const sourceIds = record(lesson.candidate).source_event_ids;
      return Array.isArray(sourceIds) && sourceIds.some((sourceId) => typeof sourceId === 'string' && eventIds.has(sourceId));
    });
    item.timing = experimentTiming(item.events as LedgerEvent[]);
  }

  return {
    summary,
    events,
    iterations: [...iterations.values()],
    memory: { planner_contexts: plannerContexts, lessons },
  };
}
