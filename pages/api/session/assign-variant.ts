import type { NextApiRequest, NextApiResponse } from 'next';
import { getCookie } from 'cookies-next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const db        = await getDb();
    const sessionId = getCookie('experiment_session_id', { req, res });

    if (!sessionId) return res.status(400).json({ error: 'No session' });

    const userIdRaw = getCookie('user_id', { req, res });
    let userId: number | null = null;

    if (userIdRaw) {
      try {
        userId = parseInt(String(userIdRaw), 10);
        if (isNaN(userId)) userId = null;
      } catch {
        userId = null;
      }
    }

    const r = Math.random();
    let newGroup = 'control';
    if      (r < 0.25) newGroup = 'control';
    else if (r < 0.50) newGroup = 'variant_a';
    else if (r < 0.75) newGroup = 'variant_b';
    else               newGroup = 'variant_c';

    await db.run(
      'UPDATE sessions SET experiment_group = ?, user_id = ? WHERE id = ?',
      [newGroup, userId, sessionId],
    );

    console.log(`[assign-variant] user=${userId} session=${sessionId} group=${newGroup}`);
    return res.status(200).json({ sessionId, experimentGroup: newGroup });
  } catch (err) {
    console.error('[session/assign-variant]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
