// pages/api/auth/logout.ts
import type { NextApiRequest, NextApiResponse } from "next";
import { deleteCookie } from "cookies-next"; // <-- Bunu ekle
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

  // 1. Mevcut temizlik fonksiyonların (Auth, User ID vs.)
  clearAuthCookie(req, res);
  clearSessionCookie(req, res);
  clearPermissionCookie(req, res);
  clearGuestCookie(req, res);

  // 2. KRİTİK EKLEME: Deney Oturumunu İmha Et
  // Bu silinmezse, sonraki kullanıcı Control yerine eski Varyantı görür.
  deleteCookie("experiment_session_id", { req, res, path: '/' });

  res.status(200).json({
    success: true,
    message: "Tüm oturum ve deney verileri temizlendi.",
  });

  console.log("Logout yapıldı: Kullanıcı ve Deney verileri sıfırlandı.");
}