import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';
import { getCookie } from 'cookies-next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const events = req.body;

  if (!Array.isArray(events)) {
    return res.status(400).json({ error: 'Request body must be an array of events' });
  }

  if (events.length === 0) {
    return res.status(200).json({ success: true, processed: 0 });
  }

  const userIdRaw = getCookie('user_id', { req, res });
  let userId: number | null = null;

  if (userIdRaw) {
    const parsed = parseInt(String(userIdRaw), 10);
    if (!isNaN(parsed)) userId = parsed;
  }

  try {
    const db   = await getDb();
    const stmt = await db.prepare(
      'INSERT INTO events (session_id, user_id, experiment_group, event_type, event_data, page_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)',
    );

    for (const event of events) {
      if (!event || typeof event !== 'object') continue;
      await stmt.run(
        event.sessionId       ?? null,
        userId,
        event.experimentGroup ?? null,
        event.eventType       ?? null,
        event.eventData ? JSON.stringify(event.eventData) : null,
        event.pageUrl         ?? null,
        event.timestamp       ?? Date.now(),
      );
    }

    await stmt.finalize();
    return res.status(200).json({ success: true, processed: events.length });
  } catch (err) {
    console.error('[events/batch]', err);
    return res.status(500).json({ error: 'Failed to process events' });
  }
}
