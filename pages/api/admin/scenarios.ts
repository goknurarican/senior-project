import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';
import { requireAdmin } from '../../../lib/auth';

export default requireAdmin(async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const db = await getDb();

    if (req.method === 'GET') {
      const scenarios = await db.all('SELECT * FROM scenarios ORDER BY id ASC');
      return res.status(200).json(scenarios);
    }

    if (req.method === 'POST') {
      const { name, type, target_page, selector, params, probability } = req.body;

      if (!name || !type) {
        return res.status(400).json({ error: 'name and type are required' });
      }

      const prob = parseFloat(probability);
      if (isNaN(prob) || prob < 0 || prob > 1) {
        return res.status(400).json({ error: 'probability must be a number between 0 and 1' });
      }

      await db.run(
        'INSERT INTO scenarios (name, type, target_page, selector, params, probability) VALUES (?, ?, ?, ?, ?, ?)',
        [name, type, target_page ?? null, selector ?? null, params ? JSON.stringify(params) : null, prob],
      );

      return res.status(201).json({ success: true });
    }

    return res.status(405).json({ error: 'Method not allowed' });
  } catch (err) {
    console.error('[admin/scenarios]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});
