// pages/api/experiment/phase.ts
import { NextApiRequest, NextApiResponse } from "next";
import getDb from "../../../lib/db";
import { getCookie } from "cookies-next";
import { assignExperimentGroup } from "../../../lib/session";
// YENİ: Çerezleri (Cookie) temizlemek için auth fonksiyonlarını çağırıyoruz
import { clearAuthCookie, clearSessionCookie, clearGuestCookie } from "../../../lib/auth";

const SESSION_COOKIE = "experiment_session_id";

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const db = await getDb();
  const sessionId = getCookie(SESSION_COOKIE, { req, res }) as string;

  if (!sessionId) {
    return res.status(400).json({ error: "No session found" });
  }

  const { action } = req.body;
  const session = await db.get("SELECT * FROM sessions WHERE id = ?", sessionId);

  if (!session) {
    return res.status(404).json({ error: "Session not found" });
  }

  if (action === "start_variant") {
    let variant = session.assigned_variant;
    if (!variant || variant === "control") {
      variant = assignExperimentGroup();
    }

    await db.run(
      `UPDATE sessions SET phase = ?, experiment_group = ? WHERE id = ?`,
      [variant, variant, sessionId]
    );

    try {
      await fetch("http://127.0.0.1:5001/send_negative_trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id:       sessionId,
          scenario_name:    `phase_change_${variant}`,
          scenario_type:    "phase_change",
          experiment_group: variant,
          phase:            variant,
          timestamp:        Date.now()
        }),
      });
    } catch (e) {}

    return res.json({ status: "success", phase: variant, group: variant });
  }

  if (action === "end_experiment") {
    await db.run(`UPDATE sessions SET phase = 'completed' WHERE id = ?`, sessionId);

    try {
      await fetch("http://127.0.0.1:5001/send_negative_trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id:       sessionId,
          scenario_name:    "experiment_end",
          scenario_type:    "experiment_end",
          experiment_group: session.experiment_group,
          phase:            "completed",
          timestamp:        Date.now()
        }),
      });
    } catch (e) {}

    // BİR SONRAKİ DENEK İÇİN SİSTEMİ SIFIRLAMA (LOGOUT)
    // Önceki deneğe ait tüm kimlik ve session verilerini tarayıcıdan siliyoruz.
    clearAuthCookie(req, res);
    clearSessionCookie(req, res);
    try { clearGuestCookie(req, res); } catch(e) {}

    return res.json({ status: "success", phase: "completed" });
  }

  return res.status(400).json({ error: "Invalid action." });
}