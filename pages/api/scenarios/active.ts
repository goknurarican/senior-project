import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { page } = req.query;
    const rawPath  = Array.isArray(page) ? page[0] : (page || '/');
    const cleanPath = rawPath.split('?')[0];

    const db = await getDb();

    const scenarios = await db.all(
      `SELECT * FROM scenarios
       WHERE (target_page = ? OR target_page = '*')
         AND enabled = 1
       ORDER BY RANDOM()`,
      [cleanPath],
    );

    return res.status(200).json(scenarios);
  } catch (err) {
    console.error('[scenarios/active]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
