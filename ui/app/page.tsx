'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

type Json = Record<string, unknown>;
type Event = { event_id?: string; seq?: number; timestamp?: string; event_type?: string; payload?: Json };
type Run = {
  run_id: string; status: string; phase: string; started_at: string | null; updated_at: string | null;
  experiments_proposed: number; best_experiment_id: string | null; best_primary_score: number | null;
  baseline_primary_score: number | null; stop_reason_code: string | null; final_experiment_id: string | null; event_count: number;
};
type Iteration = {
  experiment_id: string; plan?: Json; contexts?: Json[]; lessons?: Json[]; implementation?: Json; gate_a?: Json;
  execution_request?: Json; execution_result?: Json; gate_b?: Json; evaluation?: Json; decision?: Json;
  failures?: Json[]; recoveries?: Json[]; events?: Event[];
};
type Detail = { summary: Run; events: Event[]; iterations: Iteration[]; memory: { planner_contexts: Json[]; lessons: Json[] } };

const object = (value: unknown): Json => value && typeof value === 'object' && !Array.isArray(value) ? value as Json : {};
const text = (value: unknown, fallback = '—') => typeof value === 'string' && value ? value : fallback;
const humanize = (value: string) => value.replaceAll('_', ' ').replaceAll('.', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
const formatScore = (value: number | null) => value === null ? '—' : value.toFixed(6);
const formatTime = (value?: string | null) => value ? new Date(value).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'No events';

function JsonView({ value, empty = 'Not recorded yet.' }: { value: unknown; empty?: string }) {
  if (value == null || (Array.isArray(value) && value.length === 0)) return <p className="empty-copy">{empty}</p>;
  return <pre className="json-view">{JSON.stringify(value, null, 2)}</pre>;
}

function Plan({ plan }: { plan?: Json }) {
  if (!plan) return <p className="empty-copy">The planner has not proposed this iteration yet.</p>;
  return <div className="detail-stack">
    <div className="callout"><span>Hypothesis</span><p>{text(plan.hypothesis)}</p></div>
    <dl className="detail-list">
      <div><dt>Change</dt><dd>{text(plan.change_summary)}</dd></div>
      <div><dt>Mechanism</dt><dd>{text(plan.expected_mechanism)}</dd></div>
      <div><dt>Success criteria</dt><dd>{text(plan.success_criteria)}</dd></div>
      <div><dt>Falsification</dt><dd>{text(plan.falsification_condition)}</dd></div>
      <div><dt>Target files</dt><dd>{Array.isArray(plan.target_files) ? plan.target_files.join(', ') : '—'}</dd></div>
    </dl>
  </div>;
}

function Implementation({ value }: { value?: Json }) {
  if (!value) return <p className="empty-copy">No patch has been created for this iteration.</p>;
  const diff = typeof value.diff === 'string' ? value.diff : null;
  return <div className="detail-stack">
    <div className="implementation-meta"><span>Commit <b>{text(value.patch_commit_sha)}</b></span><span>Files <b>{Array.isArray(value.changed_files) ? value.changed_files.join(', ') : '—'}</b></span><span>Model <b>{text(value.model_id)}</b></span></div>
    {diff ? <pre className="diff-view">{diff}</pre> : <p className="empty-copy">Patch metadata is present; the diff artifact is unavailable or too large to display.</p>}
  </div>;
}

export default function Home() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [canStart, setCanStart] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [confirmStart, setConfirmStart] = useState(false);
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async (preferred?: string | null) => {
    try {
      const response = await fetch('/api/runs', { cache: 'no-store' });
      const payload = await response.json() as { runs?: Run[]; latest_run_id?: string | null; can_start?: boolean; error?: string };
      if (!response.ok) throw new Error(payload.error ?? 'Could not list runs.');
      setRuns(payload.runs ?? []); setCanStart(Boolean(payload.can_start));
      setSelectedId((current) => preferred ?? current ?? payload.latest_run_id ?? null); setError('');
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load runs.'); }
    finally { setLoading(false); }
  }, []);

  const load = useCallback(async (runId: string) => {
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, { cache: 'no-store' });
      const payload = await response.json() as Detail & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? 'Could not load the run.');
      setDetail(payload); setError('');
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load the run.'); }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);
  useEffect(() => {
    if (!selectedId) return;
    const timer = window.setTimeout(() => void load(selectedId), 0);
    return () => window.clearTimeout(timer);
  }, [load, selectedId]);
  useEffect(() => {
    const timer = window.setInterval(() => { void refresh(selectedId); if (selectedId) void load(selectedId); }, 5000);
    return () => window.clearInterval(timer);
  }, [load, refresh, selectedId]);

  const start = useCallback(async () => {
    setStarting(true); setError('');
    try {
      const response = await fetch('/api/runs/start', { method: 'POST' });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? 'Could not start the run.');
      setConfirmStart(false); setNotice('Live workflow launched. The run appears after setup and preflight create its first ledger event.');
      window.setTimeout(() => void refresh(), 1500);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not start the run.'); }
    finally { setStarting(false); }
  }, [refresh]);

  const current = detail?.summary ?? runs.find((run) => run.run_id === selectedId) ?? null;
  const delta = current?.best_primary_score != null && current.baseline_primary_score != null ? current.best_primary_score - current.baseline_primary_score : null;
  const latestIteration = detail?.iterations.at(-1)?.experiment_id;
  const memory = useMemo(() => ({ contexts: detail?.memory.planner_contexts.length ?? 0, lessons: detail?.memory.lessons.length ?? 0 }), [detail]);

  return <main className="app-shell dashboard-shell">
    <aside className="sidebar runs-sidebar">
      <div className="brand"><span className="brand-mark">T</span><span><span className="brand-name">TacoRank</span><small className="brand-subtitle">Run control</small></span></div>
      <button className="start-button" type="button" onClick={() => setConfirmStart(true)} disabled={starting}>▶ <span>Start new run</span></button>
      <div className="runs-heading"><span>All runs</span><b>{runs.length}</b></div>
      <nav className="run-list" aria-label="All TacoRank runs">
        {runs.map((run, index) => <button key={run.run_id} type="button" className={`run-list-item ${selectedId === run.run_id ? 'selected' : ''}`} onClick={() => setSelectedId(run.run_id)}><span className={`run-state-dot ${run.status}`} /><span><strong>{run.run_id}</strong><small>{humanize(run.phase)} · {formatTime(run.updated_at)}</small></span>{index === 0 && <em>Latest</em>}</button>)}
        {!loading && runs.length === 0 && <p className="empty-sidebar">No ledger-backed runs yet.</p>}
      </nav>
      <div className="sidebar-foot"><div className={`health-dot ${error ? 'failed' : 'live'}`} /><div><strong>{error ? 'API issue' : 'Repository connected'}</strong><span>Refreshes every 5 seconds</span></div></div>
    </aside>

    <section className="workspace dashboard-workspace">
      <header className="topbar dashboard-topbar"><div><p className="eyebrow">Autonomous research dashboard</p><h1>{current?.run_id ?? 'TacoRank runs'}</h1></div><button className="refresh-button" type="button" onClick={() => { void refresh(selectedId); if (selectedId) void load(selectedId); }}>↻ Refresh</button></header>
      {error && <div className="error-banner" role="alert"><span>{error}</span><button type="button" onClick={() => setError('')}>×</button></div>}
      {notice && <div className="notice-banner"><span>{notice}</span><button type="button" onClick={() => setNotice('')}>×</button></div>}

      {!current ? <section className="empty-dashboard"><span>◎</span><h2>No run selected</h2><p>Start a production run or wait for a ledger-backed run to appear.</p></section> : <>
        <section className="run-hero"><div><span className={`status-badge ${current.status}`}><i />{humanize(current.status)}</span><p className="hero-kicker">Current phase</p><h2>{humanize(current.phase)}</h2><p>Last durable evidence {formatTime(current.updated_at)} · {current.event_count} ledger events</p></div><div className="hero-score"><span>BEST PRIMARY</span><strong>{formatScore(current.best_primary_score)}</strong><small>{delta === null ? 'Baseline unavailable' : `${delta >= 0 ? '+' : ''}${delta.toFixed(6)} vs FM baseline`}</small></div></section>
        <section className="metric-grid dashboard-metrics"><article className="metric-card cyan"><span>Iterations</span><strong>{current.experiments_proposed}</strong><small>planner proposals</small></article><article className="metric-card pink"><span>Baseline</span><strong>{formatScore(current.baseline_primary_score)}</strong><small>protected FM</small></article><article className="metric-card cyan"><span>Memory</span><strong>{memory.lessons}</strong><small>{memory.contexts} planner contexts</small></article><article className="metric-card neutral"><span>Final selection</span><strong className="text-metric">{current.final_experiment_id ?? 'Pending'}</strong><small>{current.stop_reason_code ?? 'stop rule has not fired'}</small></article></section>
        <section className="section-heading"><div><p className="eyebrow">Complete research record</p><h2>All iterations</h2></div><span>{detail?.iterations.length ?? 0} total</span></section>
        <section className="iteration-list">
          {detail?.iterations.map((iteration, index) => {
            const metricSet = object(object(iteration.evaluation).metric_set); const decision = object(iteration.decision);
            return <details className="iteration-card" key={iteration.experiment_id} open={iteration.experiment_id === latestIteration}><summary><span className="iteration-index">{String(index + 1).padStart(2, '0')}</span><span><strong>{iteration.experiment_id}</strong><small>{text(iteration.plan?.family, 'Unclassified')} · {humanize(text(decision.decision, 'in progress'))}</small></span><span className="iteration-score">{typeof metricSet.primary_score === 'number' ? metricSet.primary_score.toFixed(6) : '—'}</span><span className="summary-chevron">⌄</span></summary>
              <div className="iteration-body"><section className="iteration-section"><div className="subheading"><span>01</span><h3>Plan</h3></div><Plan plan={iteration.plan} /></section><section className="iteration-section"><div className="subheading"><span>02</span><h3>Memory & context</h3></div><JsonView value={{ contexts: iteration.contexts ?? [], lessons: iteration.lessons ?? [] }} empty="No iteration memory recorded." /></section><section className="iteration-section"><div className="subheading"><span>03</span><h3>Current implementation</h3></div><Implementation value={iteration.implementation} /></section><section className="iteration-section"><div className="subheading"><span>04</span><h3>Execution & evaluation</h3></div><div className="evidence-grid"><div><h4>Gate A</h4><JsonView value={iteration.gate_a} /></div><div><h4>Execution</h4><JsonView value={iteration.execution_result ?? iteration.execution_request} /></div><div><h4>Gate B</h4><JsonView value={iteration.gate_b} /></div><div><h4>Evaluation</h4><JsonView value={iteration.evaluation} /></div></div></section>
              {(iteration.failures?.length || iteration.recoveries?.length) ? <section className="iteration-section warning-section"><div className="subheading"><span>!</span><h3>Failures & bounded recovery</h3></div><JsonView value={{ failures: iteration.failures ?? [], recoveries: iteration.recoveries ?? [] }} /></section> : null}<details className="raw-evidence"><summary>Raw ledger evidence ({iteration.events?.length ?? 0} events)</summary><JsonView value={iteration.events} /></details></div></details>;
          })}
          {detail && detail.iterations.length === 0 && <div className="empty-panel">No experiment proposed yet. Baseline and planner events are available in the ledger below.</div>}
        </section>
        <section className="two-column-panels"><section className="panel"><div className="panel-heading"><div><p className="eyebrow">Persistent memory</p><h3>Lessons</h3></div><span>{memory.lessons}</span></div><JsonView value={detail?.memory.lessons} empty="No durable lessons recorded yet." /></section><section className="panel"><div className="panel-heading"><div><p className="eyebrow">Immutable evidence</p><h3>Full event ledger</h3></div><span>{detail?.events.length ?? 0}</span></div><div className="compact-ledger">{detail?.events.slice().reverse().map((event) => <article key={event.event_id ?? event.seq}><span>#{event.seq}</span><strong>{event.event_type}</strong><time>{formatTime(event.timestamp)}</time></article>)}</div></section></section>
      </>}
    </section>

    {confirmStart && <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !starting) setConfirmStart(false); }}><section className="start-modal" role="dialog" aria-modal="true" aria-labelledby="start-title"><span className="modal-icon">▶</span><p className="eyebrow">Production workflow</p><h2 id="start-title">Start a new live run?</h2><p>This launches setup, non-mutating preflight, the paid DeepSeek + Trae loop, final submission checking, and ledger validation. It uses a fresh run ID and may run for hours.</p>{!canStart && <div className="credential-note">DEEPSEEK_API_KEY is not available to the dashboard. Export it before starting the UI.</div>}<div className="modal-actions"><button type="button" className="cancel-button" onClick={() => setConfirmStart(false)} disabled={starting}>Cancel</button><button type="button" className="start-button modal-start" onClick={() => void start()} disabled={starting || !canStart}>{starting ? 'Launching…' : 'Start production run'}</button></div></section></div>}
  </main>;
}
