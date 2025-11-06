// lib/auth.ts
import { getCookie, setCookie, deleteCookie } from 'cookies-next';
import getDb from './db';

export async function authenticate(email: string, password: string) {
  const db = await getDb();
  const user = await db.get(
    'SELECT * FROM users WHERE email = ? AND password = ?',
    [email, password]
  );
  return user;
}

export async function getUserFromCookie(req: any, res: any) {
  const userId = getCookie('user_id', { req, res });
  if (!userId) return null;
  
  const db = await getDb();
  return await db.get('SELECT * FROM users WHERE id = ?', userId);
}

export function setAuthCookie(res: any, userId: number) {
  setCookie('user_id', userId, {
    res,
    maxAge: 7 * 24 * 60 * 60, // 7 days
    httpOnly: false,
    sameSite: 'lax'
  });
}

export function clearAuthCookie(res: any) {
  deleteCookie('user_id', { res });
}

export function requireAuth(handler: any) {
  return async (req: any, res: any) => {
    const user = await getUserFromCookie(req, res);
    if (!user) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
    req.user = user;
    return handler(req, res);
  };
}

export function requireAdmin(handler: any) {
  return async (req: any, res: any) => {
    const user = await getUserFromCookie(req, res);
    if (!user || user.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }
    req.user = user;
    return handler(req, res);
  };
}
