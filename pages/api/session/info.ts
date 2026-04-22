import type { NextApiRequest, NextApiResponse } from 'next';
import { getCookie, setCookie } from 'cookies-next';
import getDb from '../../../lib/db';
import { v4 as uuidv4 } from 'uuid';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const db = await getDb();

    let sessionId        = getCookie('experiment_session_id', { req, res }) as string | undefined;
    let group            = 'control';
    let phase            = 'control';
    let assignedVariant  = 'control';

    if (sessionId) {
      const session = await db.get(
        'SELECT experiment_group, phase, assigned_variant FROM sessions WHERE id = ?',
        sessionId,
      );
      if (session) {
        group           = session.experiment_group || 'control';
        phase           = session.phase            || 'control';
        assignedVariant = session.assigned_variant || 'control';
      } else {
        sessionId = undefined;
      }
    }

    if (!sessionId) {
      sessionId = uuidv4();
      await db.run(
        'INSERT INTO sessions (id, experiment_group, phase, assigned_variant, user_agent, ip) VALUES (?, ?, ?, ?, ?, ?)',
        [sessionId, 'control', 'control', 'control',
          req.headers['user-agent'] || '',
          (req.headers['x-forwarded-for'] as string) || ''],
      );

      setCookie('experiment_session_id', sessionId, {
        req, res, maxAge: 60 * 60 * 24 * 30, path: '/',
      });
    }

    return res.status(200).json({ sessionId, experimentGroup: group, phase, assignedVariant });
  } catch (err) {
    console.error('[session/info]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
