// pages/api/scenarios/trigger.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const { sessionId, scenarioId, status } = req.body;
  const db = await getDb();
  
  await db.run(
    'INSERT INTO scenario_triggers (session_id, scenario_id, status) VALUES (?, ?, ?)',
    [sessionId, scenarioId, status]
  );
  
  res.status(200).json({ success: true });
}
