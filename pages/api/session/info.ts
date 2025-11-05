// pages/api/session/info.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import { getOrCreateSession, getSessionInfo } from '../../../lib/session';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const sessionId = await getOrCreateSession(req, res);
  const sessionInfo = await getSessionInfo(sessionId);
  
  res.status(200).json({
    sessionId: sessionId,
    experimentGroup: sessionInfo?.experiment_group || 'control'
  });
}
