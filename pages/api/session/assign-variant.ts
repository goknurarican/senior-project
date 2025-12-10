// pages/api/session/assign-variant.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import { getCookie } from 'cookies-next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).end();

  const db = await getDb();
  const sessionId = getCookie('experiment_session_id', { req, res });

  if (!sessionId) return res.status(400).json({ error: 'No session' });

  // Rastgele Varyant Seçimi (Login Olan Kullanıcı İçin)
  const r = Math.random();
  let newGroup = 'control';
  if (r < 0.25) newGroup = 'control';
  else if (r < 0.50) newGroup = 'variant_a';
  else if (r < 0.75) newGroup = 'variant_b';
  else newGroup = 'variant_c';

  // Veritabanını Güncelle
  await db.run(
    'UPDATE sessions SET experiment_group = ? WHERE id = ?',
    [newGroup, sessionId]
  );

  console.log(`🎲 Kullanıcı Login Oldu -> Yeni Varyant Atandı: ${newGroup}`);

  res.status(200).json({ sessionId, experimentGroup: newGroup });
}