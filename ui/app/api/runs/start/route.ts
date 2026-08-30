import { closeSync, openSync } from 'node:fs';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { NextResponse } from 'next/server';

import { repositoryRoot } from '@/lib/run-store';

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
        env: { ...process.env, DEEPSEEK_API_KEY: apiKey, PYTHONDONTWRITEBYTECODE: '1' },
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
