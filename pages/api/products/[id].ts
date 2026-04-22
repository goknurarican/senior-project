import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const id = parseInt(req.query.id as string, 10);

  if (isNaN(id)) {
    return res.status(400).json({ error: 'Invalid product id' });
  }

  try {
    const db = await getDb();
    const product = await db.get('SELECT * FROM products WHERE id = ?', [id]);

    if (!product) {
      return res.status(404).json({ error: 'Product not found' });
    }

    return res.status(200).json(product);
  } catch (err) {
    console.error('[products/[id]]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
