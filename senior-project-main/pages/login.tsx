// pages/login.tsx
import { useState } from "react";
import { NextRouter, useRouter } from "next/router";
import Link from "next/link";
import { getCookie } from "cookies-next";
import { userCookie } from "../types/types";

export default function Login() {
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const router: NextRouter = useRouter();

// pages/login.tsx içindeki güncel handleSubmit fonksiyonu

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      // 1. Login İsteği
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data: userCookie = await res.json();

      if (res.ok) {
        // 2. Login Başarılı - Sepeti Birleştir (Mevcut Kodun)
        try {
          const mergeRes = await fetch("/api/cart/merge", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          });
          const mergeData = await mergeRes.json();
          if (mergeData.merged && mergeData.itemCount > 0) {
            console.log(`${mergeData.itemCount} ürün sepete taşındı`);
          }
        } catch (mergeError) {
          console.error("Sepet birleştirme hatası:", mergeError);
        }

        // 👇 3. YENİ EKLENEN KISIM: VARYANT ATAMA (Randomizer) 👇
        // Kullanıcı artık içeride, "Control" modundan çıkarıp gerçek deneye alalım.
        try {
            const variantRes = await fetch('/api/session/assign-variant', { method: 'POST' });

            if (variantRes.ok) {
                const variantData = await variantRes.json();

                // SDK'ya CANLI SİNYAL gönder: "Hey, grup değişti, reload yapmadan kendini güncelle!"
                if (typeof window !== 'undefined') {
                    const event = new CustomEvent('session:update', {
                        detail: {
                            sessionId: variantData.sessionId,
                            experimentGroup: variantData.experimentGroup
                        }
                    });
                    window.dispatchEvent(event);
                    console.log(`🎲 Login sonrası yeni varyant atandı: ${variantData.experimentGroup}`);
                }
            }
        } catch (variantError) {
            console.error("Varyant atama servisi hatası:", variantError);
        }
        // 👆 VARYANT İŞLEMİ BİTTİ 👆

        // 4. Yönlendirme (Mevcut Kodun)
        const raw: string | undefined = getCookie("user_id") as string;
        const userid: string | null = raw ? JSON.parse(raw) : null;

        if (data.user?.role === "admin") {
          router.push("/admin");
        } else {
          // Deney başladı, ürünlere yolla
          router.push("/products");
        }
      } else {
        setError(data.error || "Login failed");
        setIsLoading(false);
      }
    } catch (err) {
      console.error("Login error:", err);
      setError("An error occurred. Please try again.");
      setIsLoading(false);
    }
  };
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="max-w-md w-full">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900">Login</h2>
            <p className="mt-2 text-sm text-gray-600">
              Sign in to your account
            </p>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="email-input-div">
              <label className="block text-sm font-medium text-gray-700">
                Email
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                placeholder="admin@test.com or user@test.com"
              />
            </div>

            <div className="password-input-div">
              <label className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                placeholder="admin123 or user123"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-blue-400 disabled:cursor-not-allowed"
            >
              {isLoading ? "Signing in..." : "Sign in"}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              Don't have an account?{" "}
              <Link
                href="/signup"
                className="font-medium text-blue-600 hover:text-blue-500"
              >
                Sign up here
              </Link>
            </p>
          </div>

          <div className="mt-6">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white text-gray-500">
                  Test Accounts
                </span>
              </div>
            </div>

            <div className="mt-4 space-y-2 text-sm text-gray-600">
              <div className="p-3 bg-gray-50 rounded">
                <p className="font-semibold">Admin:</p>
                <p>Email: admin@test.com</p>
                <p>Password: admin123</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="font-semibold">User:</p>
                <p>Email: user@test.com</p>
                <p>Password: user123</p>
              </div>
            </div>
          </div>

          <div className="mt-4 text-center">
            <Link
              href="/"
              className="text-sm text-blue-600 hover:text-blue-500"
            >
              Continue as guest →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}