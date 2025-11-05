// pages/api/admin/scenarios/[id].ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { id } = req.query;
  const db = await getDb();
  
  if (req.method === 'PATCH') {
    const { enabled } = req.body;
    
    await db.run(
      'UPDATE scenarios SET enabled = ? WHERE id = ?',
      [enabled ? 1 : 0, id]
    );
    
    return res.status(200).json({ success: true });
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}
