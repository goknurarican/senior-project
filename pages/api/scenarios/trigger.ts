// pages/api/scenarios/trigger.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';


function formatLocalTimestampWithMs() {
  const d = new Date();

  const year   = d.getFullYear();
  const month  = String(d.getMonth() + 1).padStart(2, '0');
  const day    = String(d.getDate()).padStart(2, '0');
  const hour   = String(d.getHours()).padStart(2, '0');
  const minute = String(d.getMinutes()).padStart(2, '0');
  const second = String(d.getSeconds()).padStart(2, '0');
  const ms     = String(d.getMilliseconds()).padStart(3, '0');

  return `${year}-${month}-${day} ${hour}:${minute}:${second}:${ms}`;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const { sessionId, scenarioId, status, details } = req.body;
  const db = await getDb();

  try {
    // 1. Resmi senaryo tetiklenme kaydı (Trigger Log)
    await db.run(
      'INSERT INTO scenario_triggers (session_id, scenario_id, status) VALUES (?, ?, ?)',
      [sessionId, scenarioId, status]
    );

    // 2. Bunu aynı zamanda genel "Event" akışına da ekleyelim (Analiz kolaylığı için)
    // Böylece: PageView -> Click -> SCENARIO_TRIGGER -> Click sıralaması görülür.
    await db.run(
      `INSERT INTO events (session_id, event_type, event_data, page_url, timestamp) 
       VALUES (?, ?, ?, ?, ?)`,
      [
        sessionId,
        'SCENARIO_TRIGGERED',
        JSON.stringify({ scenarioId, status, details }),
        req.headers.referer || '/',
        Date.now()
      ]
    );

    res.status(200).json({ success: true });
  } catch (error) {
    console.error("Trigger log hatası:", error);
    res.status(500).json({ error: 'Database error' });
  }
}