// pages/api/auth/login.ts

import type { NextApiRequest, NextApiResponse } from "next";

import {
  authenticate,
  clearGuestCookie,
  setAuthCookie,
} from "../../../lib/auth";

import { createSession } from "../../../lib/createSession";

export default async function handler(
  req: NextApiRequest,

  res: NextApiResponse
) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ error: "Email and password are required" });
  }

  try {
    const user = await authenticate(email, password);
    if (!user) {
      console.log("kullanııc db'de bulunamadı");
      return res.status(401).json({ error: "Invalid credentials" });
    }

    await createSession(req, res, user.id);

    setAuthCookie(req, res, user.id, user.role);
    clearGuestCookie(req, res);
    res.status(200).json({
      success: true,

      user: {
        id: user.id,

        email: user.email,

        name: user.name,

        role: user.role,
      },
    });

    console.log("Login başarılı - Cookie'ler set edildi");
  } catch (error) {
    console.error("Login error:", error);

    return res.status(500).json({ error: "Internal server error" });
  }
}
