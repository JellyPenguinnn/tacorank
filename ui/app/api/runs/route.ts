import { existsSync } from 'node:fs';
import path from 'node:path';
import { NextResponse } from 'next/server';

import { listRuns, repositoryRoot } from '@/lib/run-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const runs = await listRuns();
    const launcherLock = path.join(repositoryRoot(), '.tacorank', 'live-run.lock');
    return NextResponse.json({
      runs,
      latest_run_id: runs[0]?.run_id ?? null,
      can_start: !existsSync(launcherLock),
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Could not list runs.' }, { status: 500 });
  }
}
