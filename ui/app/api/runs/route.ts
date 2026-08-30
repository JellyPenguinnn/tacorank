import { existsSync } from 'node:fs';
import path from 'node:path';
import { NextResponse } from 'next/server';

import { hasActiveDashboardLaunch, listRuns, repositoryRoot } from '@/lib/run-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const runs = await listRuns();
    const tacorankDirectory = path.join(repositoryRoot(), '.tacorank');
    const launcherLock = path.join(tacorankDirectory, 'live-run.lock');
    const startLock = path.join(tacorankDirectory, 'dashboard-start.lock');
    return NextResponse.json({
      runs,
      latest_run_id: runs[0]?.run_id ?? null,
      can_start: !existsSync(launcherLock) && !existsSync(startLock) && !(await hasActiveDashboardLaunch()),
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Could not list runs.' }, { status: 500 });
  }
}
