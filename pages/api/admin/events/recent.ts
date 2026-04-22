import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../../lib/db';
import { requireAdmin } from '../../../../lib/auth';

export default requireAdmin(async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const db = await getDb();

    const events = await db.all(`
      SELECT * FROM events
      ORDER BY created_at DESC
      LIMIT 10
    `);

    return res.status(200).json(events);
  } catch (err) {
    console.error('[admin/events/recent]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});
