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

  // DEĞİŞİKLİK BURADA:
  // CASE WHEN mantığı ile:
  // 1. target_page "*" DEĞİLSE (yani özel sayfaysa) ona 0 puan ver (öne geçsin)
  // 2. target_page "*" İSE ona 1 puan ver (arkaya geçsin)
  // 3. Kendi aralarında RANDOM() sırala.
  const scenarios = await db.all(
  `
  SELECT * FROM scenarios 
  WHERE (target_page = ? OR target_page = "*")
  AND enabled = 1
  ORDER BY 
    CASE WHEN target_page = '*' THEN 1 ELSE 0 END ASC,
    RANDOM()
  `,
  [cleanPath]
);
  res.status(200).json(scenarios);
}