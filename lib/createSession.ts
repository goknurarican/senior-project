import { NextApiRequest, NextApiResponse } from "next";
import getDb from "./db";
import { ExperimentGroup } from "../types/types";
import { assignExperimentGroup } from "./session";
import { getGuestCookie, setSessionCookie } from "./auth";

export async function createSession(
  req: NextApiRequest,
  res: NextApiResponse,
  userId?: number
): Promise<{ sessionId: string; group: ExperimentGroup; phase: string; assignedVariant: string }> {
  const db = await getDb();
  const sessionId = getGuestCookie(req, res);

  let assignedVariant: ExperimentGroup = "control";

  // Admin kontrolü
  if (req.body.email === "admin@test.com" || req.body.password === "admin123") {
    assignedVariant = "control";
  } else {
    // Variant ATANIR ama henüz AKTİF DEĞİL
    assignedVariant = assignExperimentGroup();
    // Control grubuna düşerse zaten iki fazda da senaryo tetiklenmez
  }

  // phase=control ile başla, assigned_variant ve user_id sakla
  await db.run(
    `INSERT INTO sessions (id, user_id, experiment_group, phase, assigned_variant, user_agent, ip)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
    [
      sessionId,
      userId ?? null,                   // ← user_id (login'den geliyor)
      "control",                        // ← Başlangıçta CONTROL
      "control",                        // ← Phase = control
      assignedVariant,                  // ← Asıl variant saklanıyor
      req.headers["user-agent"] || "",
      req.socket.remoteAddress || "",
    ]
  );

  setSessionCookie(req, res, sessionId);

  console.log(`[Session] User signed up → assigned_variant=${assignedVariant}, phase=control`);

  return {
    sessionId,
    group: "control" as ExperimentGroup,  // ← Başta control
    phase: "control",
    assignedVariant,
  };
}