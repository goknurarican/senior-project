// pages/api/admin/events/recent.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();
  
  const events = await db.all(`
    SELECT * FROM events 
    ORDER BY created_at DESC 
    LIMIT 10
  `);
  
  res.status(200).json(events);
}
