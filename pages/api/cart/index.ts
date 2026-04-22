import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';
import { v4 } from 'uuid';
import { getGuestCookie, getUserFromCookie, setGuestCookie } from '../../../lib/auth';
import { getCookie } from 'cookies-next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const db = await getDb();

    const experiment_session_id = getCookie('experiment_session_id', { req, res });
    const user                  = await getUserFromCookie(req, res);

    let guestCookie = getGuestCookie(req, res);
    if (!guestCookie && !user) {
      guestCookie = v4();
      setGuestCookie(req, res, guestCookie);
    }

    const sessionId = user ? experiment_session_id : guestCookie;

    if (req.method === 'GET') {
      const items = await db.all(
        `SELECT cart_items.*, products.title, products.price, products.image
         FROM cart_items
         JOIN products ON cart_items.product_id = products.id
         WHERE cart_items.session_id = ?`,
        [sessionId],
      );
      return res.status(200).json(items);
    }

    if (req.method === 'POST') {
      const { productId, quantity = 1 } = req.body;

      if (!productId) {
        return res.status(400).json({ error: 'productId is required' });
      }

      const pid = parseInt(String(productId), 10);
      if (isNaN(pid)) {
        return res.status(400).json({ error: 'Invalid productId' });
      }

      const qty = Math.max(1, parseInt(String(quantity), 10) || 1);

      const existing = await db.get(
        'SELECT * FROM cart_items WHERE session_id = ? AND product_id = ?',
        [sessionId, pid],
      );

      if (existing) {
        await db.run(
          'UPDATE cart_items SET quantity = quantity + ? WHERE id = ?',
          [qty, existing.id],
        );
      } else {
        await db.run(
          'INSERT INTO cart_items (session_id, product_id, quantity) VALUES (?, ?, ?)',
          [sessionId, pid, qty],
        );
      }

      return res.status(200).json({ success: true });
    }

    if (req.method === 'DELETE') {
      const pid = parseInt(req.query.productId as string, 10);
      if (isNaN(pid)) {
        return res.status(400).json({ error: 'Invalid productId' });
      }

      await db.run(
        'DELETE FROM cart_items WHERE session_id = ? AND product_id = ?',
        [sessionId, pid],
      );
      return res.status(200).json({ success: true });
    }

    return res.status(405).json({ error: 'Method not allowed' });
  } catch (err) {
    console.error('[cart]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
