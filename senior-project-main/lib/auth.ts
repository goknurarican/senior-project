// lib/auth.ts
import { getCookie, setCookie, deleteCookie } from "cookies-next";
import getDb from "./db";
import bcrypt from "bcryptjs";

const SESSION_COOKIE = "experiment_session_id";

export async function authenticate(email: string, password: string) {
  // Bu fonksiyon çalışıyor api/auth/login'de doğrulamayı sağlıyor sorun yok.
  try {
    const db = await getDb();

    const user = await db.get("SELECT * FROM users WHERE email = ?", [email]);
    if (!user) return null;

    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) return null;

    return user;
  } catch (error) {
    console.error("[authenticate] Database error:", error);
    throw new Error("Authentication failed");
  }
}

export async function getUserFromCookie(req: any, res: any) {
  const userId = getCookie("user_id", { req, res });
  if (!userId) return null;

  const db = await getDb();
  return await db.get("SELECT * FROM users WHERE id = ?", userId);
}

export function setAuthCookie(
  req: any,
  res: any,
  userId: any,
  permission: any,
) {
  //user_id cookie'si artık çalışıyor OĞUZHAN YAPAR USTA.
  setCookie(
    "user_id",
    userId  ,
    {
      req,
      res,
      maxAge: 7 * 24 * 60 * 60, // 7 days
      httpOnly: false,
      sameSite: "lax",
    }
  );
  setCookie("user_permission", permission, {
    req,
    res,
    maxAge: 7 * 24 * 60 * 60, // 7 days
    httpOnly: false,
    sameSite: "lax",
  });
}

//guest cookie'ler

export function setGuestCookie(req, res, guest_id) {
  setCookie("guest_id", guest_id, {
    req,
    res,
    maxAge: 7 * 24 * 60 * 60,
    httpOnly: false,
    sameSite: "lax",
  });
}
export function getGuestCookie(req, res) {
  return getCookie("guest_id", { req, res });
}
export function clearGuestCookie(req, res) {
  deleteCookie("guest_id", { req, res });
}

//session cookie'ler
export function setSessionCookie(req, res, getGuestCookie) {
  setCookie(SESSION_COOKIE, getGuestCookie, {
    req,
    res,
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
    sameSite: "lax",
  });
}

export function clearAuthCookie(req: any, res: any) {
  deleteCookie("user_id", { req, res }); // bu çalışmıyordu. Oğuzhan her zamanki gibi bunu da çözdü
}
export function clearPermissionCookie(req, res) {
  deleteCookie("user_permission", { req, res }); //bunu ekledim routing sorununu çözmek için. Oğuzhan.
}
export function clearSessionCookie(req: any, res: any) {
  deleteCookie(SESSION_COOKIE, { req, res, path: "/" });
}
export function requireAuth(handler: any) {
  return async (req: any, res: any) => {
    const user = await getUserFromCookie(req, res);
    if (!user) {
      return res.status(401).json({ error: "Unauthorized" });
    }
    req.user = user;
    return handler(req, res);
  };
}

export function requireAdmin(handler: any) {
  return async (req: any, res: any) => {
    const user = await getUserFromCookie(req, res);
    if (!user || user.role !== "admin") {
      return res.status(403).json({ error: "Forbidden" });
    }
    req.user = user;
    return handler(req, res);
  };
}
