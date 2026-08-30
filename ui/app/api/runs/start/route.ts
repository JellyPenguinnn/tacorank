import { closeSync, openSync } from 'node:fs';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { NextResponse } from 'next/server';

import { repositoryRoot } from '@/lib/run-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST() {
  const root = repositoryRoot();
  if (!process.env.DEEPSEEK_API_KEY) {
    return NextResponse.json({ error: 'DEEPSEEK_API_KEY is not available to the dashboard process.' }, { status: 412 });
  }
  const lock = path.join(root, '.tacorank', 'live-run.lock');
  try {
    await fs.access(lock);
    return NextResponse.json({ error: 'A script-managed live run is already active. Open the latest run to monitor it.' }, { status: 409 });
  } catch {
    // No launcher lock is present; the reviewed script creates it atomically.
  }
  const script = path.join(root, 'run-new-live.sh');
  try {
    await fs.access(script);
    const logDirectory = path.join(root, '.tacorank', 'dashboard-launches');
    await fs.mkdir(logDirectory, { recursive: true });
    const launchId = new Date().toISOString().replaceAll(/[-:.TZ]/g, '');
    const logPath = path.join(logDirectory, `${launchId}.log`);
    const log = openSync(logPath, 'a');
    let child;
    try {
      child = spawn(script, [], {
        cwd: root,
        detached: true,
        env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
        shell: false,
        stdio: ['ignore', log, log],
      });
      child.on('error', () => { /* The reviewed script records normal failures in its launch log. */ });
      child.unref();
    } finally {
      closeSync(log);
    }
    return NextResponse.json({ status: 'started', launch_id: launchId, pid: child.pid });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Could not start the live run.' }, { status: 500 });
  }
}
