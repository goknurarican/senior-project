// pages/api/events/batch.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';
import { getCookie } from 'cookies-next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const events = req.body;
  const db = await getDb();

  // 1. Cookie'den User ID'yi çek (Eğer kullanıcı login olmuşsa bu cookie vardır)
  const userIdRaw = getCookie('user_id', { req, res });
  let userId = null;

  if (userIdRaw) {
    try {
      userId = JSON.parse(userIdRaw as string);
    } catch (e) {
      userId = userIdRaw; // JSON değilse direkt string olarak al
    }
  }

  try {
    // 2. SQL Sorgusunu Hazırla (Performance için prepare kullanılır)
    // experiment_group ve user_id sütunlarını ekledik
    const stmt = await db.prepare(
      'INSERT INTO events (session_id, user_id, experiment_group, event_type, event_data, page_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)'
    );

    for (const event of events) {
      await stmt.run(
        event.sessionId,
        userId,                 // Cookie'den gelen User ID
        event.experimentGroup,  // SDK'dan gelen o anki grup (control/variant)
        event.eventType,
        JSON.stringify(event.eventData),
        event.pageUrl,
        event.timestamp
      );
    }

    await stmt.finalize(); // İşlemi kapat
    
    res.status(200).json({ success: true, processed: events.length });
  } catch (error) {
    console.error('Event batch error:', error);
    res.status(500).json({ error: 'Failed to process events' });
  }
}