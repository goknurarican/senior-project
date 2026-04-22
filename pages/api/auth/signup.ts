// pages/api/auth/signup.ts
import type { NextApiRequest, NextApiResponse } from "next";
import bcrypt from "bcryptjs";
import { getDb } from "../../../lib/db";
import { v4 as uuidv4 } from "uuid";
import { assignExperimentGroup } from "../../../lib/session";
import {
  clearGuestCookie,
  getGuestCookie,
  setAuthCookie,
  setSessionCookie,
} from "../../../lib/auth";
import { getCookie } from "cookies-next";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { name, email, password, age, gender, handedness, vision_correction } = req.body;

  // Validation
  if (!name || !email || !password) {
    return res.status(400).json({ error: "All fields are required" });
  }

  try {
    const db = await getDb();
    // kullanıcı var mı diye kontrol ediyorum
    const existingUser = await db.get("SELECT id FROM users WHERE email = ?", [
      email,
    ]);

    if (existingUser) {
      return res.status(400).json({ error: "Email already registered" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const result = await db.run(
      `INSERT INTO users (name, email, password, role, age, gender, handedness, vision_correction)
       VALUES (?, ?, ?, 'user', ?, ?, ?, ?)`,
      [name, email, hashedPassword,
       age ? parseInt(age) : null,
       gender || null,
       handedness || 'right',
       vision_correction || 'none']
    );

    const newUser = await db.get(
      "SELECT id, name, email, role FROM users WHERE id = ?",
      [result.lastID] // sor
    );

    let experiment_session_id = getCookie('experiment_session_id', { req, res }) as string | undefined;
    let group = assignExperimentGroup();

    if (experiment_session_id) {
      // Session already exists (created by /api/session/info) — just link it to the user
      await db.run(
        `UPDATE sessions SET user_id=?, experiment_group=? WHERE id=?`,
        [newUser.id, group, experiment_session_id]
      );
    } else {
      // No session yet — create one
      experiment_session_id = uuidv4();
      await db.run(
        `INSERT INTO sessions (id, user_id, experiment_group, user_agent, ip)
         VALUES (?, ?, ?, ?, ?)`,
        [
          experiment_session_id,
          newUser.id,
          group,
          req.headers["user-agent"] || "",
          (req.socket as any).remoteAddress || "",
        ]
      );
    }
    setSessionCookie(req, res, experiment_session_id);

    // Notify trigger_server — 3 s timeout so signup never hangs if server is starting
    const _ac = new AbortController();
    const _t  = setTimeout(() => _ac.abort(), 3000);
    try {
      await fetch("http://127.0.0.1:5001/set_session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: experiment_session_id }),
        signal: _ac.signal,
      });
    } catch (_) {} finally { clearTimeout(_t); }

    setAuthCookie(req, res, newUser.id, newUser.role);
    clearGuestCookie(req, res);
    // Sonra JSON dön
    res.status(201).json({
      success: true,
      message: "Account created successfully",
      cookie: "2 cookie de eklendi. signup.ts ",
      user: {
        id: newUser.id,
        name: newUser.name,
        email: newUser.email,
        role: newUser.role,
      },
    });
  } catch (error) {
    console.error("Sign up error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
}
