import { NextResponse } from 'next/server';

import { listRuns } from '@/lib/run-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const runs = await listRuns();
    return NextResponse.json({
      runs,
      latest_run_id: runs[0]?.run_id ?? null,
      can_start: Boolean(process.env.DEEPSEEK_API_KEY),
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Could not list runs.' }, { status: 500 });
  }
}
