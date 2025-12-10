// pages/api/session/info.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import { getCookie, setCookie } from 'cookies-next';
import getDb from '../../../lib/db';
import { v4 as uuidv4 } from 'uuid';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();

  let sessionId = getCookie('experiment_session_id', { req, res });
  let group = 'control'; // Varsayılan: Her zaman Control

  // 1. Session Var mı Kontrol Et
  if (sessionId) {
    const session = await db.get('SELECT experiment_group FROM sessions WHERE id = ?', sessionId);
    if (session) {
      group = session.experiment_group;
    } else {
      sessionId = null; // Cookie var ama DB'de yoksa silinmiş demektir
    }
  }

  // 2. Yoksa Yeni Oluştur (AMA SADECE CONTROL)
  if (!sessionId) {
    sessionId = uuidv4();
    // Burada RANDOM YOK! Sadece 'control'
    group = 'control';

    await db.run(
      'INSERT INTO sessions (id, experiment_group, user_agent, ip) VALUES (?, ?, ?, ?)',
      [sessionId, group, req.headers['user-agent'] || '', req.headers['x-forwarded-for'] || '']
    );

    setCookie('experiment_session_id', sessionId, {
      req, res, maxAge: 60 * 60 * 24 * 30, path: '/'
    });
  }

  res.status(200).json({ sessionId, experimentGroup: group });
}