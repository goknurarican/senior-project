// pages/api/admin/sessions/[id]/events.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { id } = req.query;
  const db = await getDb();
  
  const events = await db.all(`
    SELECT * FROM events 
    WHERE session_id = ?
    ORDER BY timestamp ASC
  `, [id]);
  
  res.status(200).json(events);
}
