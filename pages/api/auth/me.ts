// pages/api/auth/me.ts
import type { NextApiRequest, NextApiResponse } from "next";
import { getUserFromCookie } from "../../../lib/auth";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const user = await getUserFromCookie(req, res);
  console.log("user : ", user); // Undefined dönüyor
  if (!user) {
    return res.status(401).json({ error: "Not authenticated" });
  }

  res.status(200).json({
    id: user.id, //user.id idi, şu an 1 yaptım hep sahte bir admin döndürecek 401 hatası vermemek için
    email: user.email,
    name: user.name,
    role: user.role,
  });
}
