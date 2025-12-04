// pages/api/scenarios/active.ts
import type { NextApiRequest, NextApiResponse } from "next";
import getDb from "../../../lib/db";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { page } = req.query;
  const db = await getDb();

  // Get ALL scenarios - client will filter based on enabled status
  const scenarios = await db.all(
    `
    SELECT * FROM scenarios 
    WHERE (target_page = ? OR target_page = "*")
    ORDER BY probability DESC
  `,
    [page || "/"]
  );

  // Return all scenarios without modification
  // The client SDK will handle probability adjustments
  res.status(200).json(scenarios);
}
