// pages/api/session/info.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import { getCookie, setCookie } from 'cookies-next';
import getDb from '../../../lib/db';
import { v4 as uuidv4 } from 'uuid';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();

  let sessionId = getCookie('experiment_session_id', { req, res });
  let group = 'control';
  let phase = 'control';
  let assignedVariant = 'control';

  // 1. Session Var mı Kontrol Et
  if (sessionId) {
    const session = await db.get(
      'SELECT experiment_group, phase, assigned_variant FROM sessions WHERE id = ?',
      sessionId
    );
    if (session) {
      group = session.experiment_group || 'control';
      phase = session.phase || 'control';
      assignedVariant = session.assigned_variant || 'control';
    } else {
      sessionId = null;
    }
  }

  // 2. Yoksa Yeni Oluştur (SADECE CONTROL)
  if (!sessionId) {
    sessionId = uuidv4();
    group = 'control';
    phase = 'control';

    await db.run(
      'INSERT INTO sessions (id, experiment_group, phase, assigned_variant, user_agent, ip) VALUES (?, ?, ?, ?, ?, ?)',
      [sessionId, group, phase, 'control', req.headers['user-agent'] || '', req.headers['x-forwarded-for'] || '']
    );

    setCookie('experiment_session_id', sessionId, {
      req, res, maxAge: 60 * 60 * 24 * 30, path: '/'
    });
  }

  res.status(200).json({
    sessionId,
    experimentGroup: group,
    phase,
    assignedVariant,
  });
}