import { closeSync, openSync } from 'node:fs';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn, type ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { NextResponse } from 'next/server';

import { hasActiveDashboardLaunch, repositoryRoot, writeLaunchRecord } from '@/lib/run-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_API_KEY_LENGTH = 4096;

export async function POST(request: Request) {
  const root = repositoryRoot();
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Enter a DeepSeek API key to start the run.' }, { status: 400 });
  }
  const apiKey = body && typeof body === 'object' && 'api_key' in body && typeof body.api_key === 'string'
    ? body.api_key.trim()
    : '';
  if (!apiKey) {
    return NextResponse.json({ error: 'Enter a DeepSeek API key to start the run.' }, { status: 400 });
  }
  if (apiKey.length > MAX_API_KEY_LENGTH || apiKey.includes('\0') || apiKey.includes('\n') || apiKey.includes('\r')) {
    return NextResponse.json({ error: 'The DeepSeek API key is invalid.' }, { status: 400 });
  }
  const tacorankDirectory = path.join(root, '.tacorank');
  await fs.mkdir(tacorankDirectory, { recursive: true });
  const startLock = path.join(tacorankDirectory, 'dashboard-start.lock');
  try {
    await fs.mkdir(startLock);
  } catch {
    return NextResponse.json({ error: 'Another dashboard launch is being prepared.' }, { status: 409 });
  }

  try {
    const lock = path.join(root, '.tacorank', 'live-run.lock');
    try {
      await fs.access(lock);
      return NextResponse.json({ error: 'A script-managed live run is already active. Open the latest run to monitor it.' }, { status: 409 });
    } catch {
      // No launcher lock is present; the reviewed script creates it atomically.
    }
    if (await hasActiveDashboardLaunch()) {
      return NextResponse.json({ error: 'A dashboard-launched run is already active. Open the latest run to monitor it.' }, { status: 409 });
    }
    const script = path.join(root, 'run-new-live.sh');
    try {
      await fs.access(script);
      const logDirectory = path.join(root, '.tacorank', 'dashboard-launches');
      await fs.mkdir(logDirectory, { recursive: true });
      const startedAt = new Date().toISOString();
      const launchId = startedAt.replaceAll(/[-:.TZ]/g, '');
      const runId = `run_${launchId}_${randomBytes(4).toString('hex')}`;
      const logPath = path.join(logDirectory, `${launchId}.log`);
      const launchRecord = {
        schema_version: 'tacorank.dashboard-launch.v1' as const,
        launch_id: launchId,
        run_id: runId,
        started_at: startedAt,
        pid: null,
      };
      await writeLaunchRecord(launchRecord);
      const log = openSync(logPath, 'a');
      let child: ChildProcess;
      try {
        child = spawn(script, [], {
          cwd: root,
          detached: true,
          env: {
            ...process.env,
            DEEPSEEK_API_KEY: apiKey,
            PYTHONDONTWRITEBYTECODE: '1',
            TACORANK_RUN_ID: runId,
          },
          shell: false,
          stdio: ['ignore', log, log],
        });
        await new Promise<void>((resolve, reject) => {
          child.once('spawn', resolve);
          child.once('error', reject);
        });
        await writeLaunchRecord({ ...launchRecord, pid: child.pid ?? null });
        child.unref();
      } finally {
        closeSync(log);
      }
      return NextResponse.json({ status: 'started', launch_id: launchId, run_id: runId, pid: child.pid });
    } catch (error) {
      return NextResponse.json({ error: error instanceof Error ? error.message : 'Could not start the live run.' }, { status: 500 });
    }
  } finally {
    await fs.rmdir(startLock).catch(() => undefined);
  }
}
