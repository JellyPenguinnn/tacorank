import { NextRequest, NextResponse } from 'next/server';

import { DashboardStopError, requestDashboardStop } from '@/lib/run-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest, context: { params: Promise<{ runId: string }> }) {
  try {
    const { runId } = await context.params;
    let body: unknown;
    try {
      body = await request.json();
    } catch {
      return NextResponse.json({ error: 'Confirm the exact run ID before stopping.' }, { status: 400 });
    }
    const confirmedRunId = body && typeof body === 'object' && 'confirm_run_id' in body
      ? body.confirm_run_id
      : null;
    if (confirmedRunId !== runId) {
      return NextResponse.json({ error: 'Confirm the exact run ID before stopping.' }, { status: 400 });
    }
    return NextResponse.json(await requestDashboardStop(runId));
  } catch (error) {
    const status = error instanceof DashboardStopError
      ? error.status
      : error instanceof Error && error.message === 'Invalid run ID.' ? 400 : 500;
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Could not stop the run.' }, { status });
  }
}
