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
  status: string;
  phase: string;
  started_at: string | null;
  updated_at: string | null;
  experiments_proposed: number;
  best_experiment_id: string | null;
  best_primary_score: number | null;
  baseline_primary_score: number | null;
  stop_reason_code: string | null;
  final_experiment_id: string | null;
  event_count: number;
};

const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const MAX_ARTIFACT_BYTES = 512_000;

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
  const metricSet = record(result.metric_set ?? payload.metric_set);
  return number(metricSet.primary_score ?? payload.primary_score);
}

export function summarizeRun(runId: string, events: LedgerEvent[]): RunSummary {
  let status = 'initializing';
  let phase = 'not_started';
  let proposed = 0;
  let bestId: string | null = null;
  let bestScore: number | null = null;
  let baselineScore: number | null = null;
  let stopReason: string | null = null;
  let finalId: string | null = null;

  for (const event of events) {
    const payload = record(event.payload);
    switch (event.event_type) {
      case 'run.started': phase = 'contract_verification'; break;
      case 'contract.verified': phase = 'baseline'; break;
      case 'baseline.verified':
        status = 'ready'; phase = 'planning'; baselineScore = metricScore(payload);
        bestId = text(payload.experiment_id) ?? 'baseline'; bestScore = baselineScore; break;
      case 'context.created': phase = `${text(nested(payload, 'context').role) ?? 'planner'}_context`; break;
      case 'experiment.proposed': status = 'running'; phase = 'coding'; proposed += 1; break;
      case 'patch.created': phase = 'patch_gate'; break;
      case 'patch.checked': phase = nested(payload, 'result').accepted === false ? 'recovery' : 'execution'; break;
      case 'execution.started': phase = 'running'; break;
      case 'execution.finished': phase = text(nested(payload, 'result').outcome) === 'success' ? 'output_gate' : 'recovery'; break;
      case 'adapter.failed': phase = 'recovery'; break;
      case 'recovery.decided': phase = 'recovery'; break;
      case 'output.checked': phase = nested(payload, 'result').accepted === false ? 'recovery' : 'evaluation'; break;
      case 'evaluation.completed': phase = 'decision'; break;
      case 'experiment.decided': phase = 'planning'; break;
      case 'best.updated': bestId = text(payload.experiment_id); bestScore = number(payload.primary_score); break;
      case 'run.stopped': status = 'stopped'; phase = 'stopped'; stopReason = text(payload.reason_code); break;
      case 'final.selected': status = 'finalizing'; phase = 'submission'; finalId = text(payload.experiment_id); break;
      case 'submission.checked':
        status = payload.accepted === true ? 'finalized' : 'failed'; phase = status; break;
    }
  }
  return {
    run_id: runId,
    status,
    phase,
    started_at: events[0]?.timestamp ?? null,
    updated_at: events.at(-1)?.timestamp ?? null,
    experiments_proposed: proposed,
    best_experiment_id: bestId,
    best_primary_score: bestScore,
    baseline_primary_score: baselineScore,
    stop_reason_code: stopReason,
    final_experiment_id: finalId,
    event_count: events.length,
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
  const directory = path.join(repositoryRoot(), 'runs');
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  const runs = await Promise.all(entries
    .filter((entry) => entry.isDirectory() && RUN_ID.test(entry.name))
    .map(async (entry) => {
      try { return summarizeRun(entry.name, await readEvents(entry.name)); }
      catch { return null; }
    }));
  return runs.filter((run): run is RunSummary => run !== null)
    .sort((a, b) => (b.updated_at ?? '').localeCompare(a.updated_at ?? ''));
}

export async function runDetail(runId: string): Promise<JsonRecord> {
  const events = await readEvents(runId);
  const summary = summarizeRun(runId, events);
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
    if (event.event_type === 'evaluation.completed') item.evaluation = payload.result;
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
  }

  return {
    summary,
    events,
    iterations: [...iterations.values()],
    memory: { planner_contexts: plannerContexts, lessons },
  };
}
