// pages/api/auth/logout.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import { clearAuthCookie } from '../../../lib/auth';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  clearAuthCookie(res);
  res.status(200).json({ success: true });
}
