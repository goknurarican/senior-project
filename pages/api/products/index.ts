import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const db = await getDb();
    const { category, search } = req.query;

    let query = 'SELECT * FROM products WHERE 1=1';
    const params: string[] = [];

    if (category && typeof category === 'string') {
      query += ' AND category = ?';
      params.push(category);
    }

    if (search && typeof search === 'string') {
      query += ' AND title LIKE ?';
      params.push(`%${search}%`);
    }

    const products = await db.all(query, params);
    return res.status(200).json(products);
  } catch (err) {
    console.error('[products]', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
