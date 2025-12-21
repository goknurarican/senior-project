// pages/api/cart/index.ts
import type { NextApiRequest, NextApiResponse } from "next";
import getDb from "../../../lib/db";
import { v4 } from "uuid";
import {
  getGuestCookie,
  getUserFromCookie,
  setGuestCookie,
} from "../../../lib/auth";
import { users } from "../../../types/types";
import { getCookie } from "cookies-next";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const db = await getDb();

  // 1. Experiment Session ID'yi çek
  let experiment_session_id = getCookie("experiment_session_id", { req, res });
  console.log("experiment_session_id in cart api:", experiment_session_id);

  const user: users | null | undefined = await getUserFromCookie(req, res);

  // 2. Guest Cookie kontrolü
  let guestCookie = getGuestCookie(req, res);

  if (!guestCookie && user === null) {
    guestCookie = v4();
    setGuestCookie(req, res, guestCookie);
  }

  // 3. KRİTİK DEĞİŞİKLİK: Hangi ID'nin kullanılacağına burada karar veriyoruz.
  // Kullanıcı varsa experiment id, yoksa guest cookie kullanılır.
  const sessionId = user ? experiment_session_id : guestCookie;

  if (req.method === "GET") {
    const items = await db.all(
      `
      SELECT cart_items.*, products.title, products.price, products.image 
      FROM cart_items 
      JOIN products ON cart_items.product_id = products.id
      WHERE cart_items.session_id = ?
    `,
      [sessionId] // Artık karmaşık OR mantığı yerine tek değişken
    );

    return res.status(200).json(items);
  }

  if (req.method === "POST") {
    const { productId, quantity = 1 } = req.body;
    console.log("productid : ", productId, "sessionId : ", sessionId);

    // Ürün zaten var mı kontrol et
    const existing = await db.get(
      "SELECT * FROM cart_items WHERE session_id = ? AND product_id = ?",
      [sessionId, productId]
    );

    if (existing) {
      await db.run(
        "UPDATE cart_items SET quantity = quantity + ? WHERE id = ?",
        [quantity, existing.id]
      );
    } else {
      await db.run(
        "INSERT INTO cart_items (session_id, product_id, quantity) VALUES (?, ?, ?)",
        [sessionId, productId, quantity]
      );
    }

    return res.status(200).json({ success: true });
  }

  if (req.method === "DELETE") {
    const { productId } = req.query;

    await db.run(
      "DELETE FROM cart_items WHERE session_id = ? AND product_id = ?",
      [sessionId, productId]
    );

    return res.status(200).json({ success: true });
  }

  return res.status(405).json({ error: "Method not allowed" });
}