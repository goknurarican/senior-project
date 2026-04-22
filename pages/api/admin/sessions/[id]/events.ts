import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../../../lib/db';
import { requireAdmin } from '../../../../../lib/auth';

export default requireAdmin(async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { id } = req.query;

  if (!id || typeof id !== 'string') {
    return res.status(400).json({ error: 'Session id required' });
  }

  try {
    const db = await getDb();

    const events = await db.all(`
      SELECT * FROM events
      WHERE session_id = ?
      ORDER BY timestamp ASC
    `, [id]);

    return res.status(200).json(events);
  } catch (err) {
    console.error('[admin/sessions/[id]/events]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});
