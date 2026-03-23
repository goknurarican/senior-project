import { getCookie } from "cookies-next";
import { NextApiRequest, NextApiResponse } from "next";
import getDb from "./db";

const SESSION_COOKIE = "experiment_session_id";

export async function getSession(req: NextApiRequest, res: NextApiResponse) {
  const db = await getDb();
  const sessionId = getCookie(SESSION_COOKIE, { req, res }) as string | undefined;

  if (!sessionId) {
    return { sessionId: null, group: "control", phase: "control", assignedVariant: "control" };
  }

  // BÜTÜN SIR BURADA: phase ve assigned_variant veritabanından çekiliyor
  const row = await db.get(
    "SELECT experiment_group, phase, assigned_variant FROM sessions WHERE id = ?",
    sessionId
  );

  if (row) {
    return {
      sessionId,
      group: row.experiment_group || "control",
      phase: row.phase || "control",
      assignedVariant: row.assigned_variant || "control"
    };
  }

  return { sessionId, group: "control", phase: "control", assignedVariant: "control" };
}