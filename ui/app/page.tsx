'use client';

import { type CSSProperties, type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

type JsonRecord = Record<string, unknown>;

type RunSnapshot = {
  run_id?: string;
  derived_from?: { event_id?: string; event_hash?: string };
  global?: JsonRecord;
  active_jobs?: JsonRecord[];
  resources?: JsonRecord;
};

type LedgerEvent = {
  event_id?: string;
  seq?: number;
  timestamp?: string;
  run_id?: string;
  event_type?: string;
  event_hash?: string;
  payload?: JsonRecord;
};

type DirectoryHandle = {
  name: string;
  getFileHandle(name: string): Promise<{ getFile(): Promise<File> }>;
};

type ScorePoint = { id: string; score: number };

const demoSnapshot: RunSnapshot = {
  run_id: 'run_2026_08_30',
  derived_from: { event_id: 'evt_000048' },
  global: {
    status: 'running', phase: 'evaluation', best_experiment_id: 'exp_0012', best_primary_score: 0.6438,
    baseline_primary_score: 0.6356, experiments_proposed: 12, remaining_iterations: 38, full_evaluations_completed: 7,
  },
  active_jobs: [{ experiment_id: 'exp_0012', attempt: 1, phase: 'evaluation', fidelity: 'full' }],
  resources: { elapsed_wall_time_seconds: 8077, provider_tokens: 18420, cpu_time_ms: 3240000 },
};

const demoScores = [0.6331, 0.6362, 0.6351, 0.6391, 0.6384, 0.6438];
const demoEvents: LedgerEvent[] = demoScores.map((score, index) => ({
  event_id: `evt_${String(43 + index).padStart(6, '0')}`,
  seq: 43 + index,
  timestamp: new Date(Date.now() - (demoScores.length - index - 1) * 180_000).toISOString(),
  run_id: 'run_2026_08_30',
  event_type: index === demoScores.length - 1 ? 'evaluation.completed' : index % 2 ? 'output.checked' : 'evaluation.completed',
  payload: index % 2 ? { experiment_id: `exp_${String(index * 2 + 2).padStart(4, '0')}`, accepted: true } : {
    experiment_id: `exp_${String(index * 2 + 1).padStart(4, '0')}`,
    result: { fidelity: 'full', metric_set: { primary_score: score } },
  },
}));

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {};
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function humanize(value: string): string {
  return value.replaceAll('_', ' ').replaceAll('.', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return [hours, minutes, remainder].map((part) => String(part).padStart(2, '0')).join(':');
}

function eventScore(event: LedgerEvent): number | null {
  const payload = asRecord(event.payload);
  const result = asRecord(payload.result);
  const metricSet = asRecord(result.metric_set ?? payload.metric_set);
  const score = metricSet.primary_score ?? payload.primary_score;
  return typeof score === 'number' && Number.isFinite(score) ? score : null;
}

function eventExperiment(event: LedgerEvent): string {
  const payload = asRecord(event.payload);
  const result = asRecord(payload.result);
  return asString(payload.experiment_id ?? result.experiment_id, 'run');
}

function describeEvent(event: LedgerEvent): string {
  const payload = asRecord(event.payload);
  const score = eventScore(event);
  const experiment = eventExperiment(event);
  const result = asRecord(payload.result);
  const fidelity = asString(result.fidelity ?? payload.fidelity);

  if (event.event_type === 'evaluation.completed') {
    return `${experiment}${fidelity ? ` · ${fidelity} fidelity` : ''}${score === null ? '' : ` · ${score.toFixed(4)}`}`;
  }
  if (event.event_type === 'output.checked') {
    const accepted = payload.accepted ?? asRecord(payload.receipt).accepted;
    return `${experiment} · output contract ${accepted === false ? 'rejected' : 'verified'}`;
  }
  if (event.event_type === 'patch.checked') return `${experiment} · Gate A patch and lineage verification recorded`;
  if (event.event_type === 'execution.finished') return `${experiment} · candidate execution finished`;
  if (event.event_type === 'adapter.failed') return `${experiment} · bounded recovery evidence recorded`;
  if (event.event_type === 'best.updated') return `${experiment} · new validation best selected`;
  if (event.event_type === 'submission.checked') return `Official submission contract ${payload.accepted === true ? 'accepted' : 'checked'}`;
  return `${experiment} · durable event appended to the ledger`;
}

function phaseStep(phase: string): number {
  if (['planning', 'planner_context', 'contract_verification', 'baseline'].includes(phase)) return 0;
  if (['coding', 'coder_context', 'patch_gate'].includes(phase)) return 1;
  if (['execution', 'running', 'output_gate', 'recovery'].includes(phase)) return 2;
  if (['evaluation', 'reflection', 'decision'].includes(phase)) return 3;
  if (['stopped', 'submission', 'finalized', 'failed'].includes(phase)) return 4;
  return 0;
}

function phaseDescription(phase: string, experiment: string): string {
  const descriptions: Record<string, string> = {
    planning: `The deterministic controller is preparing the next bounded proposal after ${experiment}.`,
    coding: `Trae is editing ${experiment} in an isolated worktree before Gate A verification.`,
    patch_gate: `${experiment} is undergoing protected-path, patch, and Git lineage checks.`,
    execution: `${experiment} is running in the hardened CPU-only candidate environment.`,
    running: `${experiment} is running in the hardened CPU-only candidate environment.`,
    output_gate: `${experiment} outputs are being checked for schema, order, and execution identity.`,
    evaluation: `The protected evaluator is scoring ${experiment} without exposing labels to the planner.`,
    recovery: `The controller is applying bounded recovery to ${experiment} without weakening a gate.`,
    submission: `The validation-best candidate is undergoing final label-free test inference and submission checks.`,
    finalized: `The official submission check passed and the run evidence is finalized.`,
    failed: `The run stopped on a failed invariant. Inspect the latest durable event before intervening.`,
  };
  return descriptions[phase] ?? `TacoRank is in ${humanize(phase).toLowerCase()} for ${experiment}.`;
}

async function readHandleText(directory: DirectoryHandle, name: string): Promise<string> {
  const handle = await directory.getFileHandle(name);
  return (await handle.getFile()).text();
}

function parseLedger(text: string): LedgerEvent[] {
  return text.split(/\r?\n/).filter(Boolean).map((line, index) => {
    try { return JSON.parse(line) as LedgerEvent; }
    catch { throw new Error(`events.jsonl contains invalid JSON on line ${index + 1}`); }
  });
}

export default function Home() {
  const [snapshot, setSnapshot] = useState<RunSnapshot>(demoSnapshot);
  const [events, setEvents] = useState<LedgerEvent[]>(demoEvents);
  const [source, setSource] = useState<'demo' | 'folder' | 'files'>('demo');
  const [lastSynced, setLastSynced] = useState<Date>(new Date());
  const [error, setError] = useState('');
  const [ledgerOpen, setLedgerOpen] = useState(false);
  const directoryRef = useRef<DirectoryHandle | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDirectory = useCallback(async (directory: DirectoryHandle) => {
    const [stateText, ledgerText] = await Promise.all([
      readHandleText(directory, 'state.json'),
      readHandleText(directory, 'events.jsonl'),
    ]);
    setSnapshot(JSON.parse(stateText) as RunSnapshot);
    setEvents(parseLedger(ledgerText));
    setSource('folder');
    setLastSynced(new Date());
    setError('');
  }, []);

  const connectRun = useCallback(async () => {
    const picker = (window as Window & { showDirectoryPicker?: () => Promise<DirectoryHandle> }).showDirectoryPicker;
    if (!picker) {
      fileInputRef.current?.click();
      return;
    }
    try {
      const directory = await picker.call(window);
      directoryRef.current = directory;
      await loadDirectory(directory);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return;
      setError(caught instanceof Error ? `${caught.message}. Choose a runs/<run_id> folder.` : 'Could not read that run folder.');
    }
  }, [loadDirectory]);

  const refreshRun = useCallback(async () => {
    if (!directoryRef.current) {
      await connectRun();
      return;
    }
    try { await loadDirectory(directoryRef.current); }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Refresh failed.'); }
  }, [connectRun, loadDirectory]);

  const importFiles = useCallback(async (change: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(change.target.files ?? []);
    const stateFile = files.find((file) => file.name === 'state.json');
    const ledgerFile = files.find((file) => file.name === 'events.jsonl');
    if (!stateFile || !ledgerFile) {
      setError('Select both state.json and events.jsonl from the same run.');
      return;
    }
    try {
      const [stateText, ledgerText] = await Promise.all([stateFile.text(), ledgerFile.text()]);
      setSnapshot(JSON.parse(stateText) as RunSnapshot);
      setEvents(parseLedger(ledgerText));
      setSource('files');
      setLastSynced(new Date());
      setError('');
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not import the run files.'); }
    change.target.value = '';
  }, []);

  useEffect(() => {
    if (source !== 'folder') return;
    const timer = window.setInterval(() => { void refreshRun(); }, 5000);
    return () => window.clearInterval(timer);
  }, [refreshRun, source]);

  const dashboard = useMemo(() => {
    const global = asRecord(snapshot.global);
    const resources = asRecord(snapshot.resources);
    const active = asRecord(snapshot.active_jobs?.[0]);
    const runId = asString(snapshot.run_id, events[0]?.run_id ?? 'unknown_run');
    const status = asString(global.status, 'initializing');
    const phase = asString(global.phase, 'not_started');
    const experiments = asNumber(global.experiments_proposed);
    const remaining = asNumber(global.remaining_iterations);
    const bestScore = typeof global.best_primary_score === 'number' ? global.best_primary_score : null;
    const baseline = typeof global.baseline_primary_score === 'number' ? global.baseline_primary_score : null;
    const activeExperiment = asString(active.experiment_id, asString(global.best_experiment_id, 'baseline'));
    const scorePoints: ScorePoint[] = events.flatMap((event) => {
      if (event.event_type !== 'evaluation.completed') return [];
      const result = asRecord(asRecord(event.payload).result);
      if (result.fidelity && result.fidelity !== 'full') return [];
      const score = eventScore(event);
      return score === null ? [] : [{ id: eventExperiment(event), score }];
    }).slice(-12);
    return { global, resources, runId, status, phase, experiments, remaining, bestScore, baseline, activeExperiment, scorePoints };
  }, [events, snapshot]);

  const phaseIndex = phaseStep(dashboard.phase);
  const latestEvents = [...events].sort((a, b) => (b.seq ?? 0) - (a.seq ?? 0)).slice(0, 4);
  const delta = dashboard.bestScore !== null && dashboard.baseline !== null ? dashboard.bestScore - dashboard.baseline : null;
  const statusLabel = source === 'demo' ? 'DEMO' : dashboard.status === 'running' ? 'LIVE' : dashboard.status.toUpperCase();
  const statusTone = dashboard.status === 'failed' ? 'failed' : dashboard.status === 'finalized' ? 'finalized' : source === 'demo' ? 'demo' : 'live';

  const chart = useMemo(() => {
    const scores = dashboard.scorePoints.length ? dashboard.scorePoints : dashboard.bestScore === null ? [] : [{ id: dashboard.activeExperiment, score: dashboard.bestScore }];
    const values = scores.map((point) => point.score).concat(dashboard.baseline === null ? [] : [dashboard.baseline]);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;
    const padding = Math.max((max - min) * .28, .001);
    const lower = min - padding;
    const upper = max + padding;
    const position = (score: number) => 8 + ((score - lower) / (upper - lower)) * 84;
    return { scores, lower, upper, position, baselinePosition: dashboard.baseline === null ? null : position(dashboard.baseline) };
  }, [dashboard]);

  return (
    <main className="app-shell">
      <input ref={fileInputRef} className="visually-hidden" type="file" multiple accept=".json,.jsonl" onChange={importFiles} />
      <aside className="sidebar">
        <div className="brand" aria-label="TacoRank home"><span className="brand-mark" aria-hidden="true">T</span><span className="brand-name">TacoRank</span></div>
        <nav className="main-nav" aria-label="Primary navigation">
          <a className="nav-item active" href="#overview"><span className="nav-icon">⌁</span>Overview</a>
          <a className="nav-item" href="#experiments"><span className="nav-icon">◎</span>Experiments<span className="nav-count">{dashboard.experiments}</span></a>
          <button className="nav-item nav-button" type="button" onClick={() => setLedgerOpen(true)}><span className="nav-icon">≡</span>Event ledger</button>
          <a className="nav-item" href="#resources"><span className="nav-icon">◫</span>Resources</a>
        </nav>
        <div className="sidebar-foot">
          <div className={`health-dot ${statusTone}`} />
          <div><strong>{source === 'demo' ? 'Preview data' : humanize(dashboard.status)}</strong><span>{source === 'folder' ? 'Auto-refreshing every 5s' : source === 'files' ? 'Imported snapshot' : 'Connect a run to monitor'}</span></div>
        </div>
      </aside>

      <section className="workspace" id="overview">
        <header className="topbar">
          <div><p className="eyebrow">Autonomous research monitor</p><h1>Research at a glance.</h1></div>
          <div className="topbar-actions">
            <button className="icon-button" type="button" aria-label="Refresh run data" title="Refresh run data" onClick={() => void refreshRun()}>↻</button>
            <button className="run-picker" type="button" onClick={() => void connectRun()}>
              <span className={`picker-status ${statusTone}`} />
              <span className="picker-label">{source === 'demo' ? 'Connect run folder' : dashboard.runId}</span>
              <span className="chevron">⌄</span>
            </button>
          </div>
        </header>

        {error && <div className="error-banner" role="alert"><span>{error}</span><button type="button" onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
        <p className="source-note"><span className={`source-pill ${source}`}>{source === 'demo' ? 'Preview' : source === 'folder' ? 'Folder connected' : 'Files imported'}</span>{source === 'demo' ? 'Choose a runs/<run_id> folder to read state.json and events.jsonl locally.' : `Last synced ${lastSynced.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}. Your files never leave this browser.`}</p>

        <div className="content-grid">
          <section className="hero-card">
            <div className="hero-copy">
              <div className="status-line"><span className={`live-pill ${statusTone}`}><i /> {statusLabel}</span><span>Experiment {dashboard.experiments}{dashboard.remaining ? ` of ${dashboard.experiments + dashboard.remaining}` : ''}</span></div>
              <p className="hero-kicker">Current phase</p>
              <h2>{humanize(dashboard.phase)}</h2>
              <p className="hero-description">{phaseDescription(dashboard.phase, dashboard.activeExperiment)}</p>
              <div className="phase-track" aria-label={`Current run phase: ${humanize(dashboard.phase)}`}>
                {['Plan', 'Edit', 'Execute', 'Evaluate', 'Finalize'].map((label, index) => <span key={label} className={`phase ${index < phaseIndex ? 'complete' : index === phaseIndex ? 'current' : ''}`} />)}
              </div>
              <div className="phase-labels">{['Plan', 'Edit', 'Execute', 'Evaluate', 'Finalize'].map((label) => <span key={label}>{label}</span>)}</div>
            </div>
            <div className="score-orbit" aria-label={`Best primary score ${dashboard.bestScore?.toFixed(4) ?? 'not available'}`}>
              <div className="orbit-ring orbit-cyan" /><div className="orbit-ring orbit-pink" />
              <div className="score-core"><span>BEST</span><strong>{dashboard.bestScore?.toFixed(4) ?? '—'}</strong><small>validation score</small></div>
            </div>
          </section>

          <section className="metric-grid" aria-label="Run metrics" id="resources">
            <article className="metric-card cyan"><span>Experiments</span><strong>{dashboard.experiments}</strong><small>{dashboard.remaining} remaining</small></article>
            <article className="metric-card pink"><span>Full evaluations</span><strong>{asNumber(dashboard.global.full_evaluations_completed)}</strong><small>trusted full-fidelity results</small></article>
            <article className="metric-card cyan"><span>Best score</span><strong>{dashboard.bestScore?.toFixed(4) ?? '—'}</strong><small>{delta === null ? 'baseline unavailable' : `${delta >= 0 ? '+' : ''}${delta.toFixed(4)} vs baseline`}</small></article>
            <article className="metric-card neutral"><span>Wall time</span><strong>{formatDuration(asNumber(dashboard.resources.elapsed_wall_time_seconds))}</strong><small>{asNumber(dashboard.resources.provider_tokens).toLocaleString()} provider tokens</small></article>
          </section>

          <section className="panel score-panel" id="experiments">
            <div className="panel-heading">
              <div><p className="eyebrow">Search trajectory</p><h3>Full-fidelity score</h3></div>
              <div className="legend"><span><i className="legend-best" /> Candidate</span><span><i className="legend-baseline" /> Baseline</span></div>
            </div>
            {chart.scores.length ? (
              <div className="dynamic-chart" role="img" aria-label={`${chart.scores.length} full-fidelity validation scores from ${chart.lower.toFixed(4)} to ${chart.upper.toFixed(4)}`}>
                <div className="chart-scale"><span>{chart.upper.toFixed(3)}</span><span>{((chart.upper + chart.lower) / 2).toFixed(3)}</span><span>{chart.lower.toFixed(3)}</span></div>
                <div className="chart-plot">
                  <div className="chart-grid grid-top" /><div className="chart-grid grid-middle" /><div className="chart-grid grid-bottom" />
                  {chart.baselinePosition !== null && <div className="dynamic-baseline" style={{ '--baseline-y': `${chart.baselinePosition}%` } as CSSProperties}><span>FM {dashboard.baseline?.toFixed(4)}</span></div>}
                  <div className="score-columns">
                    {chart.scores.map((point, index) => <div className="score-column" key={`${point.id}-${index}`}><span className="point-stem" style={{ '--point-y': `${chart.position(point.score)}%` } as CSSProperties} /><span className="point-dot" style={{ '--point-y': `${chart.position(point.score)}%` } as CSSProperties} title={`${point.id}: ${point.score.toFixed(4)}`} /><small>{index === 0 || index === chart.scores.length - 1 || index % 3 === 0 ? point.id.replace('exp_', '') : ''}</small></div>)}
                  </div>
                </div>
              </div>
            ) : <div className="empty-chart">Full-fidelity scores will appear after evaluation.</div>}
          </section>

          <section className="panel event-panel" id="events">
            <div className="panel-heading"><div><p className="eyebrow">Immutable evidence</p><h3>Latest events</h3></div><button className="text-button" type="button" onClick={() => setLedgerOpen(true)}>View ledger →</button></div>
            <div className="event-list">
              {latestEvents.map((event) => <article className="event-row" key={event.event_id ?? event.seq}><div className={`event-glyph ${event.event_type?.includes('failed') ? 'pink' : event.event_type?.includes('evaluation') ? 'cyan' : 'neutral'}`}>{event.event_type?.includes('failed') ? '!' : '✓'}</div><div className="event-copy"><div><span className="event-id">#{String(event.seq ?? '').padStart(4, '0')}</span><strong>{event.event_type ?? 'event'}</strong></div><p>{describeEvent(event)}</p></div><time>{event.timestamp ? new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}</time></article>)}
            </div>
          </section>
        </div>
      </section>

      {ledgerOpen && <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setLedgerOpen(false); }}>
        <aside className="ledger-drawer" role="dialog" aria-modal="true" aria-labelledby="ledger-title">
          <div className="drawer-heading"><div><p className="eyebrow">Read-only event stream</p><h2 id="ledger-title">Event ledger</h2></div><button type="button" onClick={() => setLedgerOpen(false)} aria-label="Close ledger">×</button></div>
          <div className="drawer-summary"><span>{events.length} events</span><span>Head {snapshot.derived_from?.event_id ?? latestEvents[0]?.event_id ?? '—'}</span></div>
          <div className="drawer-events">{[...events].reverse().map((event) => <article key={event.event_id ?? event.seq}><div><span>#{String(event.seq ?? '').padStart(6, '0')}</span><time>{event.timestamp ? new Date(event.timestamp).toLocaleString() : ''}</time></div><strong>{event.event_type}</strong><p>{describeEvent(event)}</p></article>)}</div>
        </aside>
      </div>}
    </main>
  );
}
