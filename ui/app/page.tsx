'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

type Json = Record<string, unknown>;
type Event = { event_id?: string; seq?: number; timestamp?: string; event_type?: string; payload?: Json };
type Run = {
  run_id: string; source: 'ledger' | 'launch'; status: string; phase: string; is_live: boolean; launch_error: string | null;
  can_stop: boolean; stop_requested_at: string | null;
  started_at: string | null; updated_at: string | null;
  experiments_proposed: number; best_experiment_id: string | null; best_primary_score: number | null;
  best_primary_fidelity: string | null;
  baseline_primary_score: number | null; stop_reason_code: string | null; final_experiment_id: string | null; event_count: number;
  current_experiment_id: string | null; current_attempt: number | null; current_fidelity: string | null;
  stage_started_at: string | null; configured_timeout_seconds: number | null; estimated_deadline: string | null;
  last_event_id: string | null; last_event_type: string | null; last_event_at: string | null;
};
type Iteration = {
  experiment_id: string; plan?: Json; contexts?: Json[]; lessons?: Json[]; implementation?: Json; gate_a?: Json;
  execution_request?: Json; execution_result?: Json; gate_b?: Json; evaluation?: Json; decision?: Json;
  failures?: Json[]; recoveries?: Json[]; events?: Event[];
  timing?: Json;
};
type Detail = { summary: Run; events: Event[]; iterations: Iteration[]; memory: { planner_contexts: Json[]; lessons: Json[] } };
type CampaignId = 'objective_temporal_50' | 'standard';

const object = (value: unknown): Json => value && typeof value === 'object' && !Array.isArray(value) ? value as Json : {};
const text = (value: unknown, fallback = '—') => typeof value === 'string' && value ? value : fallback;
const humanize = (value: string) => value.replaceAll('_', ' ').replaceAll('.', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
const formatScore = (value: number | null) => value === null ? '—' : value.toFixed(6);
const formatTime = (value?: string | null) => value ? new Date(value).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'No events';
const formatDuration = (value: number | null) => {
  if (value === null || !Number.isFinite(value)) return '—';
  const seconds = Math.max(0, Math.floor(value));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return hours > 0
    ? `${hours}h ${String(minutes).padStart(2, '0')}m ${String(remainder).padStart(2, '0')}s`
    : `${minutes}m ${String(remainder).padStart(2, '0')}s`;
};
const elapsedSince = (value: string | null | undefined, now: number) => value ? Math.max(0, (now - new Date(value).getTime()) / 1000) : null;
const parentExperimentId = (plan?: Json) => typeof plan?.parent_experiment_id === 'string' && plan.parent_experiment_id ? plan.parent_experiment_id : null;
const lineageLabel = (plan?: Json) => {
  const parent = parentExperimentId(plan);
  if (!parent) return 'Parent not recorded';
  return parent === 'baseline' ? 'Starts from baseline' : `Continues from ${parent}`;
};

const PROGRESS_STEPS = ['Plan', 'Trae coding', 'Gate A', 'Execute', 'Gate B', 'Evaluate', 'Decide', 'Finalize'];
function progressIndex(phase: string, lastEventType: string | null, failureStage?: string): number {
  if (['planning', 'planner_context', 'contract_verification', 'baseline'].includes(phase)) return 0;
  if (['coding', 'coder_context'].includes(phase)) return 1;
  if (phase === 'patch_gate') return 2;
  if (['execution', 'running'].includes(phase)) return 3;
  if (phase === 'output_gate') return 4;
  if (phase === 'evaluation') return 5;
  if (phase === 'decision') return 6;
  if (['stopped', 'submission', 'finalized', 'failed'].includes(phase)) return 7;
  if (phase === 'recovery') {
    if (failureStage === 'coding') return 1;
    if (failureStage === 'patch_gate') return 2;
    if (failureStage === 'execution') return 3;
    if (failureStage === 'output_gate') return 4;
    if (failureStage === 'evaluation') return 5;
    if (lastEventType === 'patch.checked') return 2;
    if (lastEventType === 'execution.finished') return 3;
    if (lastEventType === 'output.checked') return 4;
    if (lastEventType === 'evaluation.completed') return 5;
  }
  return 0;
}

function JsonView({ value, empty = 'Not recorded yet.' }: { value: unknown; empty?: string }) {
  if (value == null || (Array.isArray(value) && value.length === 0)) return <p className="empty-copy">{empty}</p>;
  return <pre className="json-view">{JSON.stringify(value, null, 2)}</pre>;
}

function Plan({ plan }: { plan?: Json }) {
  if (!plan) return <p className="empty-copy">The planner has not proposed this iteration yet.</p>;
  const papers = Array.isArray(plan.literature_evidence)
    ? plan.literature_evidence.map(object).filter((paper) => typeof paper.title === 'string')
    : [];
  return <div className="detail-stack">
    <div className="callout"><span>Hypothesis</span><p>{text(plan.hypothesis)}</p></div>
    <dl className="detail-list">
      <div><dt>Lineage</dt><dd>{lineageLabel(plan)}</dd></div>
      <div><dt>Research direction</dt><dd>{humanize(text(plan.family, 'unclassified'))}</dd></div>
      <div><dt>Change</dt><dd>{text(plan.change_summary)}</dd></div>
      <div><dt>Mechanism</dt><dd>{text(plan.expected_mechanism)}</dd></div>
      <div><dt>Success criteria</dt><dd>{text(plan.success_criteria)}</dd></div>
      <div><dt>Falsification</dt><dd>{text(plan.falsification_condition)}</dd></div>
      {plan.variant_id ? <div><dt>Campaign slot</dt><dd>{text(plan.variant_id)}</dd></div> : null}
      <div><dt>Target files</dt><dd>{Array.isArray(plan.target_files) ? plan.target_files.join(', ') : '—'}</dd></div>
    </dl>
    {plan.variant_instruction ? <div className="callout"><span>Adaptive variant</span><p>{text(plan.variant_instruction)}</p></div> : null}
    {plan.variant_parameters ? <JsonView value={plan.variant_parameters} /> : null}
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
  const [confirmStop, setConfirmStop] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [campaignId, setCampaignId] = useState<CampaignId>('objective_temporal_50');
  const [now, setNow] = useState(() => Date.now());

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
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const start = useCallback(async () => {
    setStarting(true); setError('');
    try {
      const response = await fetch('/api/runs/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey, campaign_id: campaignId }),
      });
      const payload = await response.json() as { error?: string; run_id?: string };
      if (!response.ok) throw new Error(payload.error ?? 'Could not start the run.');
      if (!payload.run_id) throw new Error('The launcher did not return its reserved run ID.');
      setConfirmStart(false); setDetail(null); setSelectedId(payload.run_id);
      setNotice(`${payload.run_id} was created. Setup and preflight progress will update here until its ledger starts.`);
      void refresh(payload.run_id);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not start the run.'); }
    finally { setApiKey(''); setStarting(false); }
  }, [apiKey, campaignId, refresh]);

  const stop = useCallback(async (runId: string) => {
    setStopping(true); setError('');
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm_run_id: runId }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? 'Could not stop the run.');
      setConfirmStop(null);
      setNotice(`Stop requested for ${runId}. The last durable ledger evidence will be preserved.`);
      await refresh(runId);
      await load(runId);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not stop the run.'); }
    finally { setStopping(false); }
  }, [load, refresh]);

  const selectedDetail = detail?.summary.run_id === selectedId ? detail : null;
  const current = selectedDetail?.summary ?? runs.find((run) => run.run_id === selectedId) ?? null;
  const delta = current?.best_primary_score != null && current.baseline_primary_score != null ? current.best_primary_score - current.baseline_primary_score : null;
  const latestIteration = selectedDetail?.iterations.at(-1)?.experiment_id;
  const currentIteration = selectedDetail?.iterations.find((iteration) => iteration.experiment_id === current?.current_experiment_id) ?? selectedDetail?.iterations.at(-1);
  const currentParent = parentExperimentId(currentIteration?.plan);
  const currentTiming = object(currentIteration?.timing);
  const terminalLoopTime = typeof currentTiming.loop_time_seconds === 'number' ? currentTiming.loop_time_seconds : null;
  const clockAnchor = current && !current.is_live && current.updated_at ? new Date(current.updated_at).getTime() : now;
  const iterationElapsed = terminalLoopTime ?? elapsedSince(typeof currentTiming.proposed_at === 'string' ? currentTiming.proposed_at : null, clockAnchor);
  const stageElapsed = elapsedSince(current?.stage_started_at, clockAnchor);
  const lastEventAge = elapsedSince(current?.last_event_at, now);
  const deadlineDelta = current?.is_live && current.estimated_deadline ? (new Date(current.estimated_deadline).getTime() - now) / 1000 : null;
  const latestFailure = currentIteration?.failures?.at(-1);
  const activeProgress = current ? progressIndex(current.phase, current.last_event_type, typeof latestFailure?.failure_stage === 'string' ? latestFailure.failure_stage : undefined) : 0;
  const memory = useMemo(() => ({ contexts: selectedDetail?.memory.planner_contexts.length ?? 0, lessons: selectedDetail?.memory.lessons.length ?? 0 }), [selectedDetail]);

  return <main className="app-shell dashboard-shell">
    <aside className="sidebar runs-sidebar">
      <div className="brand"><span className="brand-mark">T</span><span><span className="brand-name">TacoRank</span><small className="brand-subtitle">Run control</small></span></div>
      <button className="start-button" type="button" onClick={() => { setApiKey(''); setCampaignId('objective_temporal_50'); setConfirmStart(true); }} disabled={starting}>▶ <span>Start new run</span></button>
      <div className="runs-heading"><span>All runs</span><b>{runs.length}</b></div>
      <nav className="run-list" aria-label="All TacoRank runs">
        {runs.map((run, index) => <button key={run.run_id} type="button" className={`run-list-item ${selectedId === run.run_id ? 'selected' : ''}`} onClick={() => setSelectedId(run.run_id)}><span className={`run-state-dot ${run.status}`} /><span><strong>{run.run_id}</strong><small>{humanize(run.phase)} · {formatTime(run.updated_at)}</small></span>{index === 0 && <em>Latest</em>}</button>)}
        {!loading && runs.length === 0 && <p className="empty-sidebar">No runs yet.</p>}
      </nav>
      <div className="sidebar-foot"><div className={`health-dot ${error ? 'failed' : 'live'}`} /><div><strong>{error ? 'API issue' : 'Repository connected'}</strong><span>Refreshes every 5 seconds</span></div></div>
    </aside>

    <section className="workspace dashboard-workspace">
      <header className="topbar dashboard-topbar"><div><p className="eyebrow">Autonomous research dashboard</p><h1>{current?.run_id ?? 'TacoRank runs'}</h1></div><div className="topbar-actions">{current && (current.can_stop || current.status === 'stopping') && <button className="stop-button" type="button" onClick={() => setConfirmStop(current.run_id)} disabled={stopping || current.status === 'stopping'}>{current.status === 'stopping' ? '■ Stopping…' : '■ Stop run'}</button>}<button className="refresh-button" type="button" onClick={() => { void refresh(selectedId); if (selectedId) void load(selectedId); }}>↻ Refresh</button></div></header>
      {error && <div className="error-banner" role="alert"><span>{error}</span><button type="button" onClick={() => setError('')}>×</button></div>}
      {notice && <div className="notice-banner"><span>{notice}</span><button type="button" onClick={() => setNotice('')}>×</button></div>}

      {!current ? <section className="empty-dashboard"><span>◎</span><h2>No run selected</h2><p>Start a production run or wait for a ledger-backed run to appear.</p></section> : <>
        <section className="run-hero"><div><span className={`status-badge ${current.status}`}><i />{humanize(current.status)}</span><p className="hero-kicker">Current phase</p><h2>{humanize(current.phase)}</h2><p>{current.source === 'launch' ? `Launch reserved ${formatTime(current.started_at)} · waiting for the first ledger event` : `Last durable evidence ${formatTime(current.updated_at)} · ${current.event_count} ledger events`}</p></div><div className="hero-score"><span>BEST PRIMARY</span><strong>{formatScore(current.best_primary_score)}</strong><small>{current.best_experiment_id === null ? 'Waiting for the verified baseline' : `${current.best_experiment_id}${current.best_primary_fidelity ? ` · ${humanize(current.best_primary_fidelity)}` : ''}${delta === null ? '' : ` · ${delta >= 0 ? '+' : ''}${delta.toFixed(6)} vs FM`}`}</small></div></section>
        {current.launch_error && <div className="launch-error" role="alert"><strong>Setup failed before the ledger started.</strong><span>{current.launch_error}</span></div>}
        <section className="runtime-panel" aria-label="Current iteration runtime and progress">
          <div className="runtime-grid">
            <article><span>Current experiment</span><strong>{current.current_experiment_id ?? (current.source === 'launch' ? 'Preparing run' : 'Between iterations')}</strong><small>{current.current_attempt ? `Attempt ${current.current_attempt}` : current.source === 'launch' ? 'Setup and preflight' : 'No active attempt'}{current.current_fidelity ? ` · ${humanize(current.current_fidelity)}` : ''}</small>{currentIteration?.plan && <small className={`runtime-lineage ${currentParent === 'baseline' ? 'baseline' : 'continuation'}`}>{lineageLabel(currentIteration.plan)}</small>}</article>
            <article><span>Iteration runtime</span><strong>{formatDuration(iterationElapsed)}</strong><small>{terminalLoopTime !== null ? 'Exact proposal-to-terminal' : current.source === 'launch' ? 'Waiting for first proposal' : current.is_live ? 'Live since proposal' : 'Frozen at last durable event'}</small></article>
            <article><span>Phase runtime</span><strong>{formatDuration(stageElapsed)}</strong><small>{current.is_live ? `Started ${formatTime(current.stage_started_at)}` : `Frozen at ${formatTime(current.updated_at)}`}</small></article>
            <article><span>Stage timeout</span><strong>{current.configured_timeout_seconds === null ? 'Not configured' : formatDuration(current.configured_timeout_seconds)}</strong><small>{current.status === 'interrupted' ? `Deadline passed ${formatTime(current.estimated_deadline)}` : deadlineDelta === null ? 'No stage deadline' : deadlineDelta >= 0 ? `${formatDuration(deadlineDelta)} remaining` : `${formatDuration(Math.abs(deadlineDelta))} overdue`}</small></article>
            <article><span>Last ledger event</span><strong>{current.last_event_type ?? (current.source === 'launch' ? 'Waiting' : '—')}</strong><small>{current.last_event_id ? `${current.last_event_id} · ${formatDuration(lastEventAge)} ago` : 'Ledger has not started yet'}</small></article>
          </div>
          {current.source === 'launch' ? <div className={`launch-progress ${current.status}`}><i>{['failed', 'interrupted'].includes(current.status) ? '!' : current.status === 'stopping' ? '■' : '…'}</i><div><span>{current.status === 'stopping' ? 'Stopping setup' : ['failed', 'interrupted'].includes(current.status) ? 'Setup stopped' : 'Setup & preflight'}</span><small>{current.status === 'stopping' ? 'Waiting for the launcher and active worker to exit.' : current.status === 'interrupted' ? 'Stopped by the operator; no ledger evidence was fabricated.' : current.status === 'failed' ? 'Review the launcher error above, then start a fresh run.' : 'The reserved run will switch to ledger data automatically.'}</small></div></div> : <><div className="progress-heading"><div><span>Current progress</span><strong>{current.status === 'stopping' ? 'Stopping' : current.status === 'interrupted' ? 'Interrupted' : current.phase === 'recovery' ? 'Bounded recovery' : PROGRESS_STEPS[activeProgress]}</strong></div><small>{current.status === 'stopping' ? `${humanize(current.phase)} · stop requested` : current.status === 'interrupted' ? `${humanize(current.phase)} · no active controller` : `Step ${activeProgress + 1} of ${PROGRESS_STEPS.length} · ${humanize(current.phase)}`}</small></div><div className="progress-track">
            {PROGRESS_STEPS.map((step, index) => <div key={step} className={`progress-step ${index < activeProgress ? 'complete' : index === activeProgress ? current.status === 'interrupted' || current.phase === 'recovery' ? 'recovery' : 'active' : ''}`}><i>{index < activeProgress ? '✓' : index + 1}</i><span>{step}</span></div>)}
          </div></>}
        </section>
        <section className="metric-grid dashboard-metrics"><article className="metric-card cyan"><span>Iterations</span><strong>{current.experiments_proposed}</strong><small>planner proposals</small></article><article className="metric-card pink"><span>Baseline</span><strong>{formatScore(current.baseline_primary_score)}</strong><small>protected FM</small></article><article className="metric-card cyan"><span>Memory</span><strong>{memory.lessons}</strong><small>{memory.contexts} planner contexts</small></article><article className="metric-card neutral"><span>Final selection</span><strong className="text-metric">{current.final_experiment_id ?? 'Pending'}</strong><small>{current.stop_reason_code ?? 'stop rule has not fired'}</small></article></section>
        <section className="section-heading"><div><p className="eyebrow">Complete research record</p><h2>All iterations</h2></div><span>{selectedDetail?.iterations.length ?? 0} total</span></section>
        <section className="iteration-list">
          {selectedDetail?.iterations.map((iteration, index) => {
            const evaluation = object(iteration.evaluation); const metricSet = object(evaluation.metric_set); const decision = object(iteration.decision);
            const reportedScore = typeof evaluation.reported_primary_score === 'number' ? evaluation.reported_primary_score : metricSet.primary_score;
            const timing = object(iteration.timing);
            const parent = parentExperimentId(iteration.plan);
            const loopTime = typeof timing.loop_time_seconds === 'number' ? timing.loop_time_seconds : elapsedSince(typeof timing.proposed_at === 'string' ? timing.proposed_at : null, clockAnchor);
            return <details className="iteration-card" key={iteration.experiment_id} open={iteration.experiment_id === latestIteration}><summary><span className="iteration-index">{String(index + 1).padStart(2, '0')}</span><span><strong>{iteration.experiment_id}</strong><small>{text(iteration.plan?.family, 'Unclassified')} · {humanize(text(decision.decision, 'in progress'))}</small></span><span className="iteration-metrics"><span><small>Loop time</small><b>{formatDuration(loopTime)}</b></span><span><small>Score</small><b>{typeof reportedScore === 'number' ? reportedScore.toFixed(6) : '—'}</b></span></span><span className="summary-chevron">⌄</span></summary>
              <div className="iteration-body"><section className="iteration-section timing-section"><div className="subheading"><span>00</span><h3>Timing</h3></div><div className="timing-breakdown"><div><span>Proposal → terminal</span><strong>{formatDuration(loopTime)}</strong><small>{timing.terminal_at ? 'Exact terminal duration' : current.is_live ? 'Live' : 'Frozen at last durable event'}</small></div><div><span>Trae coding</span><strong>{formatDuration(typeof timing.trae_coding_time_seconds === 'number' ? timing.trae_coding_time_seconds : null)}</strong></div><div><span>Execution</span><strong>{formatDuration(typeof timing.execution_time_seconds === 'number' ? timing.execution_time_seconds : null)}</strong></div><div><span>Recovery</span><strong>{formatDuration(typeof timing.recovery_time_seconds === 'number' ? timing.recovery_time_seconds : null)}</strong></div></div></section><section className="iteration-section"><div className="subheading"><span>01</span><h3>Plan</h3></div><Plan plan={iteration.plan} /></section><section className="iteration-section"><div className="subheading"><span>02</span><h3>Memory & context</h3></div><JsonView value={{ contexts: iteration.contexts ?? [], lessons: iteration.lessons ?? [] }} empty="No iteration memory recorded." /></section><section className="iteration-section"><div className="subheading"><span>03</span><h3>Current implementation</h3></div><Implementation value={iteration.implementation} /></section><section className="iteration-section"><div className="subheading"><span>04</span><h3>Execution & evaluation</h3></div><div className="evidence-grid"><div><h4>Gate A</h4><JsonView value={iteration.gate_a} /></div><div><h4>Execution</h4><JsonView value={iteration.execution_result ?? iteration.execution_request} /></div><div><h4>Gate B</h4><JsonView value={iteration.gate_b} /></div><div><h4>Evaluation</h4><JsonView value={iteration.evaluation} /></div></div></section>
              {(iteration.failures?.length || iteration.recoveries?.length) ? <section className="iteration-section warning-section"><div className="subheading"><span>!</span><h3>Failures & bounded recovery</h3></div><JsonView value={{ failures: iteration.failures ?? [], recoveries: iteration.recoveries ?? [] }} /></section> : null}<details className="raw-evidence"><summary>Raw ledger evidence ({iteration.events?.length ?? 0} events)</summary><JsonView value={iteration.events} /></details></div></details>;
          })}
          {selectedDetail && selectedDetail.iterations.length === 0 && <div className="empty-panel">{current.source === 'launch' ? 'The run identity is reserved. Setup and preflight are running before the first ledger event.' : 'No experiment proposed yet. Baseline and planner events are available in the ledger below.'}</div>}
        </section>
        <section className="two-column-panels"><section className="panel"><div className="panel-heading"><div><p className="eyebrow">Persistent memory</p><h3>Lessons</h3></div><span>{memory.lessons}</span></div><JsonView value={selectedDetail?.memory.lessons} empty="No durable lessons recorded yet." /></section><section className="panel"><div className="panel-heading"><div><p className="eyebrow">Immutable evidence</p><h3>Full event ledger</h3></div><span>{selectedDetail?.events.length ?? 0}</span></div><div className="compact-ledger">{selectedDetail?.events.slice().reverse().map((event) => <article key={event.event_id ?? event.seq}><span>#{event.seq}</span><strong>{event.event_type}</strong><time>{formatTime(event.timestamp)}</time></article>)}</div></section></section>
      </>}
    </section>

      {confirmStart && <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !starting) { setApiKey(''); setConfirmStart(false); } }}><section className="start-modal" role="dialog" aria-modal="true" aria-labelledby="start-title"><form onSubmit={(event) => { event.preventDefault(); void start(); }}><span className="modal-icon">▶</span><p className="eyebrow">Production workflow</p><h2 id="start-title">Start a new live run?</h2><p>This launches setup, non-mutating preflight, the paid DeepSeek + Trae loop, final submission checking, and ledger validation. It uses a fresh run ID and may run for hours.</p><label className="api-key-field"><span>Research campaign</span><select value={campaignId} onChange={(event) => setCampaignId(event.target.value as CampaignId)} disabled={starting || !canStart}><option value="objective_temporal_50">Features first + objective + history · 50 max</option><option value="standard">Standard autonomous search</option></select></label><label className="api-key-field"><span>DeepSeek API key</span><input type="password" name="deepseek-api-key" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Paste your API key" autoComplete="off" autoCapitalize="none" spellCheck={false} autoFocus required disabled={starting || !canStart} /><small>The key is used only for this launch. It is not saved in browser storage, run metadata, or API responses.</small></label>{!canStart && <div className="credential-note">A script-managed live run is already active. Open the latest run to monitor it.</div>}<div className="modal-actions"><button type="button" className="cancel-button" onClick={() => { setApiKey(''); setConfirmStart(false); }} disabled={starting || !canStart}>Cancel</button><button type="submit" className="start-button modal-start" disabled={starting || !canStart || !apiKey.trim()}>{starting ? 'Launching…' : 'Start production run'}</button></div></form></section></div>}
    {confirmStop && <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !stopping) setConfirmStop(null); }}><section className="start-modal stop-modal" role="dialog" aria-modal="true" aria-labelledby="stop-title"><form onSubmit={(event) => { event.preventDefault(); void stop(confirmStop); }}><span className="modal-icon stop-icon">■</span><p className="eyebrow">Interrupt live workflow</p><h2 id="stop-title">Stop this run?</h2><p>This interrupts the controller and active worker for <code>{confirmStop}</code>. Durable ledger evidence is preserved, but the run is not finalized and may not be resumable from its current phase.</p><div className="stop-warning"><strong>No result will be fabricated.</strong><span>The dashboard will mark the operational run interrupted after its process exits.</span></div><div className="modal-actions"><button type="button" className="cancel-button" onClick={() => setConfirmStop(null)} disabled={stopping}>Keep running</button><button type="submit" className="stop-button modal-stop" disabled={stopping}>{stopping ? 'Stopping…' : 'Stop this run'}</button></div></form></section></div>}
  </main>;
}
