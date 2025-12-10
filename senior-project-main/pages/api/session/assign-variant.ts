// pages/api/session/assign-variant.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import { getCookie } from 'cookies-next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).end();

  const db = await getDb();
  const sessionId = getCookie('experiment_session_id', { req, res });

  // Login sırasında set edilen user_id cookie'sini al
  const userIdRaw = getCookie('user_id', { req, res });
  const userId = userIdRaw ? JSON.parse(userIdRaw as string) : null;

  if (!sessionId) return res.status(400).json({ error: 'No session' });

  // Rastgele Varyant Seç
  const r = Math.random();
  let newGroup = 'control';
  if (r < 0.25) newGroup = 'control';
  else if (r < 0.50) newGroup = 'variant_a';
  else if (r < 0.75) newGroup = 'variant_b';
  else newGroup = 'variant_c';

  // Veritabanını Güncelle: Hem grubu değiştir, hem de User ID'yi bağla
  await db.run(
    'UPDATE sessions SET experiment_group = ?, user_id = ? WHERE id = ?',
    [newGroup, userId, sessionId]
  );

  console.log(`🎲 Kullanıcı (ID:${userId}) Login Oldu -> Yeni Varyant: ${newGroup}`);

  res.status(200).json({ sessionId, experimentGroup: newGroup });
}