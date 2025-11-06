// lib/session.ts
import { getCookie, setCookie } from 'cookies-next';
import { v4 as uuidv4 } from 'uuid';
import getDb from './db';

export async function getOrCreateSession(req: any, res: any) {
  let sessionId = getCookie('session_id', { req, res }) as string;
  
  if (!sessionId) {
    sessionId = uuidv4();
    setCookie('session_id', sessionId, { 
      req, 
      res, 
      maxAge: 30 * 24 * 60 * 60 // 30 days
    });
    
    // Assign experiment group
    const group = assignExperimentGroup();
    
    const db = await getDb();
    await db.run(
      'INSERT INTO sessions (id, experiment_group, user_agent, ip) VALUES (?, ?, ?, ?)',
      [sessionId, group, req.headers['user-agent'] || '', req.headers['x-forwarded-for'] || req.socket.remoteAddress]
    );
  }
  
  return sessionId;
}

export function assignExperimentGroup() {
  const rand = Math.random() * 100;
  
  // 25% Control (no scenarios)
  if (rand < 25) return 'control';
  
  // 25% Variant A (Low tier - 30% probability)
  if (rand < 50) return 'variant_a';
  
  // 25% Variant B (Medium tier - 60% probability)  
  if (rand < 75) return 'variant_b';
  
  // 25% Variant C (Full tier - 100% probability)
  return 'variant_c';
}

export async function getSessionInfo(sessionId: string) {
  const db = await getDb();
  return await db.get('SELECT * FROM sessions WHERE id = ?', sessionId);
}

export async function logEvent(sessionId: string, eventType: string, eventData: any, pageUrl: string) {
  const db = await getDb();
  const timestamp = Date.now();
  
  // Get session start time for relative_t_ms
  const session = await db.get('SELECT created_at FROM sessions WHERE id = ?', sessionId);
  const sessionStart = new Date(session.created_at).getTime();
  const relativeTime = timestamp - sessionStart;
  
  await db.run(
    'INSERT INTO events (session_id, event_type, event_data, page_url, timestamp, relative_t_ms) VALUES (?, ?, ?, ?, ?, ?)',
    [sessionId, eventType, JSON.stringify(eventData), pageUrl, timestamp, relativeTime]
  );
}
