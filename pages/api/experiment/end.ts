import type { NextApiRequest, NextApiResponse } from "next";
import { deleteCookie } from "cookies-next";
import getDb from "../../../lib/db";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {


  if (req.method !== "POST") {
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    const {
      sessionId,
      triggeredScenarios = [],
      mouseData = [],
      eyeData = [],
      finishedAt = Date.now(),
      pageUrl = "",
    } = req.body;

    if (!sessionId) {
      return res.status(400).json({
        ok: false,
        message: "sessionId gerekli",
      });
    }

    const db = await getDb();

    // Session var mı kontrol et
    const session = await db.get(
      `SELECT * FROM sessions WHERE id = ?`,
      [sessionId]
    );

    if (!session) {
      return res.status(404).json({
        ok: false,
        message: "Session bulunamadı",
      });
    }

    // 1) Aktif senaryoları ended olarak kaydet
    for (const scenarioId of triggeredScenarios) {
      await db.run(
        `
        INSERT INTO scenario_triggers (session_id, scenario_id, status, triggered_at)
        VALUES (?, ?, ?, ?)
        `,
        [sessionId, scenarioId, "ended", finishedAt]
      );
    }

    // 2) Deney bitti event'i
    await db.run(
      `
      INSERT INTO events (
        session_id,
        user_id,
        experiment_group,
        phase,
        event_type,
        event_data,
        page_url,
        timestamp,
        relative_t_ms
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `,
      [
        sessionId,
        session.user_id ?? null,
        session.experiment_group ?? null,
        session.phase ?? null,
        "EXPERIMENT_FINISHED",
        JSON.stringify({
          status: "finished",
          finishedAt,
          triggeredScenarioCount: triggeredScenarios.length,
          mouseSampleCount: mouseData.length,
          eyeSampleCount: eyeData.length,
        }),
        pageUrl || null,
        finishedAt,
        null,
      ]
    );

    // 3) Mouse tracking flush
    if (mouseData.length > 0) {
      await db.run(
        `
        INSERT INTO events (
          session_id,
          user_id,
          experiment_group,
          phase,
          event_type,
          event_data,
          page_url,
          timestamp,
          relative_t_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
        [
          sessionId,
          session.user_id ?? null,
          session.experiment_group ?? null,
          session.phase ?? null,
          "MOUSE_TRACKING_FLUSH",
          JSON.stringify(mouseData),
          pageUrl || null,
          finishedAt,
          null,
        ]
      );
    }

    // 4) Eye tracking flush
    if (eyeData.length > 0) {
      await db.run(
        `
        INSERT INTO events (
          session_id,
          user_id,
          experiment_group,
          phase,
          event_type,
          event_data,
          page_url,
          timestamp,
          relative_t_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
        [
          sessionId,
          session.user_id ?? null,
          session.experiment_group ?? null,
          session.phase ?? null,
          "EYE_TRACKING_FLUSH",
          JSON.stringify(eyeData),
          pageUrl || null,
          finishedAt,
          null,
        ]
      );
    }

    // 5) Cookie temizle
    deleteCookie("experiment_session_id", { req, res, path: "/" });

    return res.status(200).json({
      ok: true,
      message: "Deney başarıyla kapatıldı",
    });
  } catch (error) {
    console.error("experiment/end error:", error);

    return res.status(500).json({
      ok: false,
      message: "Deney kapatılırken hata oluştu",
    });
  }
}