// pages/api/scenarios/active.ts
import type { NextApiRequest, NextApiResponse } from "next";
import getDb from "../../../lib/db";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { page } = req.query;

  const rawPath = Array.isArray(page) ? page[0] : (page || "/");
  const cleanPath = rawPath.split('?')[0];

  const db = await getDb();

  // DEĞİŞİKLİK: 'ORDER BY RANDOM()' kullanarak listeyi her seferinde karıştırıyoruz.
  // Böylece şans faktörü adil dağılıyor.
  const scenarios = await db.all(
    `
    SELECT * FROM scenarios 
    WHERE (target_page = ? OR target_page = "*")
    AND enabled = 1
    ORDER BY RANDOM()
  `,
    [cleanPath]
  );
  res.status(200).json(scenarios);
}