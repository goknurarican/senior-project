// pages/api/events/batch.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const events = req.body;
  const db = await getDb();
  
  try {
    for (const event of events) {
      await db.run(
        'INSERT INTO events (session_id, event_type, event_data, page_url, timestamp) VALUES (?, ?, ?, ?, ?)',
        [event.sessionId, event.eventType, JSON.stringify(event.eventData), event.pageUrl, event.timestamp]
      );
    }
    
    res.status(200).json({ success: true, processed: events.length });
  } catch (error) {
    console.error('Event batch error:', error);
    res.status(500).json({ error: 'Failed to process events' });
  }
}