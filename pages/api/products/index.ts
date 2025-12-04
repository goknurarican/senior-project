// pages/api/products/index.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();
  const { category, search } = req.query;
  
  let query = 'SELECT * FROM products WHERE 1=1';
  const params = [];
  
  if (category) {
    query += ' AND category = ?';
    params.push(category);
  }
  
  if (search) {
    query += ' AND title LIKE ?';
    params.push(`%${search}%`);
  }
  
  const products = await db.all(query, params);
  res.status(200).json(products);
}
