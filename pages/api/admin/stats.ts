import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';
import { requireAdmin } from '../../../lib/auth';

export default requireAdmin(async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const db = await getDb();

    const totalSessions  = await db.get('SELECT COUNT(*) as count FROM sessions');
    const totalEvents    = await db.get('SELECT COUNT(*) as count FROM events');
    const activeScenarios = await db.get('SELECT COUNT(*) as count FROM scenarios WHERE enabled = 1');
    const triggeredToday = await db.get(`
      SELECT COUNT(*) as count FROM scenario_triggers
      WHERE date(triggered_at) = date('now')
    `);

    return res.status(200).json({
      totalSessions:    totalSessions.count,
      totalEvents:      totalEvents.count,
      activeScenarios:  activeScenarios.count,
      triggeredToday:   triggeredToday.count,
    });
  } catch (err) {
    console.error('[admin/stats]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});
