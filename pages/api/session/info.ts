// pages/api/session/info.ts
import type { NextApiRequest, NextApiResponse } from "next";
import {  getSession } from "../../../lib/getSession";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  try {
    const { sessionId, group } = await getSession(req, res);
    res.status(200).json({
      sessionId,
      experimentGroup: group,
    });
  } catch (err) {
    console.error("[/api/session/info] ERROR:", err);

    // 500 yerine kontrollü fallback → SDK çökmesin
    res.status(200).json({
      sessionId: null,
      experimentGroup: "control",
      error: "fallback_from_session_info",
    });
  }
}
