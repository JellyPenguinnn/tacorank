import { NextRequest, NextResponse } from 'next/server';

import { runDetail } from '@/lib/run-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(_request: NextRequest, context: { params: Promise<{ runId: string }> }) {
  try {
    const { runId } = await context.params;
    return NextResponse.json(await runDetail(runId));
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Could not read run.';
    return NextResponse.json({ error: message }, { status: message === 'Invalid run ID.' ? 400 : 404 });
  }
}
