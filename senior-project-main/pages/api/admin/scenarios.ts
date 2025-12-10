// pages/api/admin/scenarios.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();
  
  if (req.method === 'GET') {
    const scenarios = await db.all('SELECT * FROM scenarios');
    return res.status(200).json(scenarios);
  }
  
  if (req.method === 'POST') {
    const { name, type, target_page, selector, params, probability } = req.body;
    
    await db.run(
      'INSERT INTO scenarios (name, type, target_page, selector, params, probability) VALUES (?, ?, ?, ?, ?, ?)',
      [name, type, target_page, selector, JSON.stringify(params), probability]
    );
    
    return res.status(201).json({ success: true });
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}
