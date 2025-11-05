// pages/api/admin/stats.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();
  
  const totalSessions = await db.get('SELECT COUNT(*) as count FROM sessions');
  const totalEvents = await db.get('SELECT COUNT(*) as count FROM events');
  const activeScenarios = await db.get('SELECT COUNT(*) as count FROM scenarios WHERE enabled = 1');
  const triggeredToday = await db.get(`
    SELECT COUNT(*) as count FROM scenario_triggers 
    WHERE date(triggered_at) = date('now')
  `);
  
  res.status(200).json({
    totalSessions: totalSessions.count,
    totalEvents: totalEvents.count,
    activeScenarios: activeScenarios.count,
    triggeredToday: triggeredToday.count
  });
}
