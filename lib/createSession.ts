import { NextApiRequest, NextApiResponse } from "next";
import getDb from "./db";
import { ExperimentGroup } from "../types/types";
import { assignExperimentGroup } from "./session";
import { setSessionCookie } from "./auth";
import { getCookie } from "cookies-next";
import { v4 as uuidv4 } from "uuid";

export async function createSession(
  req: NextApiRequest,
  res: NextApiResponse,
  userId?: number
): Promise<{ sessionId: string; group: ExperimentGroup; phase: string; assignedVariant: string }> {
  const db = await getDb();
  const existingSessionId = getCookie('experiment_session_id', { req, res }) as string | undefined;
  const sessionId = existingSessionId || uuidv4();

  let assignedVariant: ExperimentGroup = "control";

  // Admin kontrolü
  if (req.body.email === "admin@test.com" || req.body.password === "admin123") {
    assignedVariant = "control";
  } else {
    // Variant ATANIR ama henüz AKTİF DEĞİL
    assignedVariant = assignExperimentGroup();
    // Control grubuna düşerse zaten iki fazda da senaryo tetiklenmez
  }

  // Session zaten varsa (browsing sırasında /api/session/info tarafından oluşturulmuş)
  // sadece user_id ve assigned_variant güncelle, yeni INSERT yapma
  const existing = await db.get('SELECT id FROM sessions WHERE id = ?', [sessionId]);

  if (existing) {
    await db.run(
      `UPDATE sessions SET user_id=?, assigned_variant=? WHERE id=?`,
      [userId ?? null, assignedVariant, sessionId]
    );
  } else {
    await db.run(
      `INSERT INTO sessions (id, user_id, experiment_group, phase, assigned_variant, user_agent, ip)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [
        sessionId,
        userId ?? null,
        "control",
        "control",
        assignedVariant,
        req.headers["user-agent"] || "",
        req.socket.remoteAddress || "",
      ]
    );
  }

  setSessionCookie(req, res, sessionId);

  // Notify trigger_server so eye data is tagged from this moment on
  // 3 s timeout — trigger server may still be starting up when first participant logs in
  const _ac1 = new AbortController();
  const _t1  = setTimeout(() => _ac1.abort(), 3000);
  try {
    await fetch("http://127.0.0.1:5001/set_session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
      signal: _ac1.signal,
    });
  } catch (_) {} finally { clearTimeout(_t1); }

  console.log(`[Session] User signed up → assigned_variant=${assignedVariant}, phase=control`);

  return {
    sessionId,
    group: "control" as ExperimentGroup,  // ← Başta control
    phase: "control",
    assignedVariant,
  };
}