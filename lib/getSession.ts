import { getCookie } from "cookies-next";
import { NextApiRequest, NextApiResponse } from "next";
import getDb from "./db";
import { ExperimentGroup } from "../types/types";
const SESSION_COOKIE = "experiment_session_id";

export async function getSession(
  req: NextApiRequest,
  res: NextApiResponse
): Promise<{ sessionId: string | null; group: ExperimentGroup }> {
  const db = await getDb();

  const sessionId = getCookie(SESSION_COOKIE, { req, res }) as
    | string
    | undefined;

  if (!sessionId) {
    return { sessionId: null, group: "control" };
  }

  const row = await db.get(
    "SELECT experiment_group FROM sessions WHERE id = ?",
    sessionId
  );

  if (row && row.experiment_group) {
    return { sessionId, group: row.experiment_group as ExperimentGroup };
  }

  // Cookie var ama DB'de karşılığı yok → create yok, sadece control dön
  return { sessionId, group: "control" };
}
