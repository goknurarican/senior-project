// pages/api/scenarios/active.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { page } = req.query;
  const db = await getDb();
  
  const scenarios = await db.all(
    'SELECT * FROM scenarios WHERE enabled = 1 AND (target_page = ? OR target_page = "*")',
    [page]
  );
  
  res.status(200).json(scenarios);
}
