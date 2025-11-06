// pages/api/admin/sessions.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();
  
  const sessions = await db.all(`
    SELECT * FROM sessions 
    ORDER BY created_at DESC 
    LIMIT 50
  `);
  
  res.status(200).json(sessions);
}
