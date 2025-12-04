// pages/api/auth/logout.ts
import type { NextApiRequest, NextApiResponse } from "next";
import {
  clearAuthCookie,
  clearGuestCookie,
  clearPermissionCookie,
  clearSessionCookie,
} from "../../../lib/auth";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  clearAuthCookie(req, res);
  clearSessionCookie(req, res);
  clearPermissionCookie(req, res);
  clearGuestCookie(req, res);
  res.status(200).json({
    success: true,
    message: "Cookie logout.ts tarafında temizlendi",
  });

  console.log("logout yapıldı. logout.ts dosyası");
}
