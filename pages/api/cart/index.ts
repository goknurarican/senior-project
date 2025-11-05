// pages/api/cart/index.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';
import { getOrCreateSession } from '../../../lib/session';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const sessionId = await getOrCreateSession(req, res);
  const db = await getDb();
  
  if (req.method === 'GET') {
    const items = await db.all(`
      SELECT c.*, p.title, p.price, p.image 
      FROM cart_items c
      JOIN products p ON c.product_id = p.id
      WHERE c.session_id = ?
    `, [sessionId]);
    
    return res.status(200).json(items);
  }
  
  if (req.method === 'POST') {
    const { productId, quantity = 1 } = req.body;
    
    // Check if item exists
    const existing = await db.get(
      'SELECT * FROM cart_items WHERE session_id = ? AND product_id = ?',
      [sessionId, productId]
    );
    
    if (existing) {
      await db.run(
        'UPDATE cart_items SET quantity = quantity + ? WHERE id = ?',
        [quantity, existing.id]
      );
    } else {
      await db.run(
        'INSERT INTO cart_items (session_id, product_id, quantity) VALUES (?, ?, ?)',
        [sessionId, productId, quantity]
      );
    }
    
    return res.status(200).json({ success: true });
  }
  
  if (req.method === 'DELETE') {
    const { productId } = req.query;
    
    await db.run(
      'DELETE FROM cart_items WHERE session_id = ? AND product_id = ?',
      [sessionId, productId]
    );
    
    return res.status(200).json({ success: true });
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}
