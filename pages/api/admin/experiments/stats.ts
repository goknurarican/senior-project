import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../../lib/db';
import { requireAdmin } from '../../../../lib/auth';

export default requireAdmin(async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const db = await getDb();

    const totalSessions = await db.get('SELECT COUNT(*) as count FROM sessions');
    const totalEvents   = await db.get('SELECT COUNT(*) as count FROM events');

    const groups = await db.all(`
      SELECT
        s.experiment_group                                    AS group_name,
        COUNT(DISTINCT s.id)                                  AS session_count,
        COUNT(e.id)                                           AS event_count,
        ROUND(COUNT(e.id) * 1.0 / MAX(COUNT(DISTINCT s.id), 1), 2) AS avg_events,
        COUNT(DISTINCT CASE
          WHEN e.event_type = 'ADD_TO_CART' THEN e.session_id
        END)                                                  AS cart_adds,
        COUNT(DISTINCT st.id)                                 AS scenarios_triggered
      FROM sessions s
      LEFT JOIN events e           ON s.id = e.session_id
      LEFT JOIN scenario_triggers st ON s.id = st.session_id
      GROUP BY s.experiment_group
    `);

    return res.status(200).json({
      overall: {
        totalSessions:  totalSessions.count,
        totalEvents:    totalEvents.count,
        avgDuration:    0,
        conversionRate: 0,
        totalRevenue:   0,
      },
      groups,
    });
  } catch (err) {
    console.error('[admin/experiments/stats]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});
