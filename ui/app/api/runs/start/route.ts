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
const CAMPAIGNS = {
  objective_temporal_50: 'research/campaigns/objective_temporal_50.json',
  standard: null,
} as const;

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
  const campaignId = body && typeof body === 'object' && 'campaign_id' in body && typeof body.campaign_id === 'string'
    ? body.campaign_id
    : 'standard';
  if (!apiKey) {
    return NextResponse.json({ error: 'Enter a DeepSeek API key to start the run.' }, { status: 400 });
  }
  if (apiKey.length > MAX_API_KEY_LENGTH || apiKey.includes('\0') || apiKey.includes('\n') || apiKey.includes('\r')) {
    return NextResponse.json({ error: 'The DeepSeek API key is invalid.' }, { status: 400 });
  }
  if (!(campaignId in CAMPAIGNS)) {
    return NextResponse.json({ error: 'The selected research campaign is invalid.' }, { status: 400 });
  }
  const campaignPath = CAMPAIGNS[campaignId as keyof typeof CAMPAIGNS];
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
    const windows = process.platform === 'win32';
    const script = path.join(root, windows ? 'run-new-live.ps1' : 'run-new-live.sh');
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
        stop_requested_at: null,
      };
      await writeLaunchRecord(launchRecord);
      let child: ChildProcess;
      const childEnvironment = {
        ...process.env,
        DEEPSEEK_API_KEY: apiKey,
        PYTHONDONTWRITEBYTECODE: '1',
        TACORANK_RUN_ID: runId,
        TACORANK_RESEARCH_CAMPAIGN: campaignPath ?? undefined,
      };
      if (windows) {
        // A detached Windows child cannot reliably inherit Node's file handle:
        // PowerShell exits before the script starts and leaves an empty log.
        // Let PowerShell own redirection, while keeping all generated paths out
        // of the fixed command string so they cannot become shell syntax.
        await fs.writeFile(logPath, '', { encoding: 'utf8', flag: 'a', mode: 0o600 });
        const powershellCommand = [
          '$code=0',
          'try {',
          '& $env:TACORANK_LAUNCH_SCRIPT -RunId $env:TACORANK_RUN_ID -DownloadData *>&1 | Out-File -LiteralPath $env:TACORANK_LAUNCH_LOG -Append -Encoding utf8',
          '$code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }',
          '} catch {',
          '($_ | Out-String) | Out-File -LiteralPath $env:TACORANK_LAUNCH_LOG -Append -Encoding utf8',
          '$code=1',
          '}',
          'exit $code',
        ].join('; ');
        const encodedCommand = Buffer.from(powershellCommand, 'utf16le').toString('base64');
        // PowerShell 5 can exit before evaluating an encoded command when a
        // detached Node child has all three standard handles set to NUL.
        // cmd.exe supplies valid handles and waits for PowerShell, while the
        // encoded, fixed command avoids quoting any user-controlled value.
        child = spawn('cmd.exe', [
          '/d',
          '/s',
          '/c',
          `powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand ${encodedCommand}`,
        ], {
          cwd: root,
          detached: true,
          windowsHide: true,
          env: {
            ...childEnvironment,
          },
          shell: false,
          stdio: 'ignore',
        });
      } else {
        const log = openSync(logPath, 'a');
        try {
          child = spawn(script, [], {
            cwd: root,
            detached: true,
            windowsHide: true,
            env: childEnvironment,
            shell: false,
            stdio: ['ignore', log, log],
          });
        } finally {
          closeSync(log);
        }
      }
      try {
        await new Promise<void>((resolve, reject) => {
          child.once('spawn', resolve);
          child.once('error', reject);
        });
        await writeLaunchRecord({ ...launchRecord, pid: child.pid ?? null });
        child.unref();
      } catch (error) {
        if (child.pid) {
          try {
            process.kill(child.pid);
          } catch {
            // The failed child already exited.
          }
        }
        throw error;
      }
      return NextResponse.json({ status: 'started', launch_id: launchId, run_id: runId, campaign_id: campaignId, pid: child.pid });
    } catch (error) {
      return NextResponse.json({ error: error instanceof Error ? error.message : 'Could not start the live run.' }, { status: 500 });
    }
  } finally {
    await fs.rmdir(startLock).catch(() => undefined);
  }
}
