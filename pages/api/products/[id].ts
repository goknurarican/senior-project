// pages/api/products/[id].ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { id } = req.query;
  const db = await getDb();
  
  if (req.method === 'GET') {
    const product = await db.get('SELECT * FROM products WHERE id = ?', id);
    
    if (!product) {
      return res.status(404).json({ error: 'Product not found' });
    }
    
    return res.status(200).json(product);
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}
