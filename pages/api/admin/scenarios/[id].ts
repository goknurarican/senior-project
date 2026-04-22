import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../../lib/db';
import { requireAdmin } from '../../../../lib/auth';

export default requireAdmin(async function handler(req: NextApiRequest, res: NextApiResponse) {
  const id = parseInt(req.query.id as string, 10);

  if (isNaN(id)) {
    return res.status(400).json({ error: 'Invalid scenario id' });
  }

  try {
    const db = await getDb();

    if (req.method === 'PATCH') {
      const { enabled } = req.body;

      await db.run('UPDATE scenarios SET enabled = ? WHERE id = ?', [enabled ? 1 : 0, id]);
      return res.status(200).json({ success: true });
    }

    if (req.method === 'DELETE') {
      await db.run('DELETE FROM scenarios WHERE id = ?', [id]);
      return res.status(200).json({ success: true });
    }

    return res.status(405).json({ error: 'Method not allowed' });
  } catch (err) {
    console.error('[admin/scenarios/[id]]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});
