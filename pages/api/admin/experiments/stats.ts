// pages/api/admin/experiments/stats.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();
  
  // Overall stats
  const totalSessions = await db.get('SELECT COUNT(*) as count FROM sessions');
  const totalEvents = await db.get('SELECT COUNT(*) as count FROM events');
  
  // Group-wise stats
  const groups = await db.all(`
    SELECT 
      s.experiment_group as group_name,
      COUNT(DISTINCT s.id) as session_count,
      COUNT(e.id) as event_count,
      ROUND(COUNT(e.id) * 1.0 / COUNT(DISTINCT s.id), 2) as avg_events,
      COUNT(DISTINCT CASE WHEN e.event_type = 'click' AND e.event_data LIKE '%Add to Cart%' THEN e.session_id END) as cart_adds,
      COUNT(DISTINCT st.id) as scenarios_triggered
    FROM sessions s
    LEFT JOIN events e ON s.id = e.session_id
    LEFT JOIN scenario_triggers st ON s.id = st.session_id
    GROUP BY s.experiment_group
  `);
  
  res.status(200).json({
    overall: {
      totalSessions: totalSessions.count,
      totalEvents: totalEvents.count,
      avgDuration: 0, // Can calculate from events
      conversionRate: 0,
      totalRevenue: 0
    },
    groups: groups
  });
}
