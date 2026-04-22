// pages/api/scenarios/trigger.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { sessionId, scenarioId, status, details } = req.body;

  if (!sessionId || !scenarioId || !status) {
    return res.status(400).json({ error: 'sessionId, scenarioId and status are required' });
  }

  const now = Date.now();

  try {
    const db = await getDb();
    // 1. Trigger Log (Artık buraya da zamanı biz veriyoruz)
    // NOT: db.ts dosyasında scenario_triggers tablosunda 'triggered_at' sütunu integer olmalı.
    // Eğer TIMESTAMP type ise yine de Date.now() integer'ını kabul eder.
    await db.run(
      'INSERT INTO scenario_triggers (session_id, scenario_id, status, triggered_at) VALUES (?, ?, ?, ?)',
      [sessionId, scenarioId, status, now]
    );

    // 2. Events Log
    await db.run(
      `INSERT INTO events (session_id, event_type, event_data, page_url, timestamp) 
       VALUES (?, ?, ?, ?, ?)`,
      [
        sessionId,
        'SCENARIO_TRIGGERED',
        JSON.stringify({ scenarioId, status, details }),
        req.headers.referer || '/',
        now // Aynı 'now' değişkeni
      ]
    );

    res.status(200).json({ success: true });
  } catch (error) {
    console.error("Trigger log hatası:", error);
    res.status(500).json({ error: 'Database error' });
  }
}