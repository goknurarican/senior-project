import Link from "next/link";
import React, { useEffect, useState } from "react";
import { useCartStore } from "../storage/zustand";
import { useRouter } from "next/router";

function Navbar() {
  const cartCount = useCartStore((state) => state.cartItem);

  const setCartCount = useCartStore((state) => state.setCartItem);

  const [sessionInfo, setSessionInfo] = useState<any>(null);
  const router = useRouter();
  const [userinfo, setUserinfo] = useState<{
    name: string;
    role: string;
  } | null>(null);

  const { pathname, query } = router;
  const category = query.category as string | undefined;

  const isProductsPage = pathname === "/products";

  const isAllProductsActive = isProductsPage && !category;
  const isElectronicsActive = isProductsPage && category === "electronics";
  const isHomeActive = isProductsPage && category === "home";

  const handleLogout = async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    setUserinfo(null);
    router.push("/");
    window.location.reload();
  };


useEffect(() => {
  // İlk yüklemede verileri çek
  fetchCartCount();
  fetchSessionInfo();
  fetchUser();

  // YENİ EKLENEN KISIM: Event Listener
  // 'cart:refresh' sinyali gelirse fetchCartCount'u tekrar çalıştır.
  const handleCartRefresh = () => {
    console.log("[Navbar] Sepet yenileme isteği alındı, sayı güncelleniyor...");
    fetchCartCount();
  };

  window.addEventListener("cart:refresh", handleCartRefresh);

  // Cleanup (Component kapanırsa dinleyiciyi kaldır)
  return () => {
    window.removeEventListener("cart:refresh", handleCartRefresh);
  };
}, [router.asPath]); // router.asPath bağımlılığı kalabilir
  const fetchCartCount = async () => {
    const res = await fetch("/api/cart");
    const items = await res.json();
    setCartCount(
      items.reduce((acc: number, item: any) => acc + item.quantity, 0)
    );
  };

  const fetchSessionInfo = async () => {
    // getorcreatesession metodu ile session üretiyor. group ile session id üretip session cookie'sine atıyor.
    const res = await fetch("/api/session/info");
    const data = await res.json();
    setSessionInfo(data);
    if (typeof window !== 'undefined') {
        const event = new CustomEvent('session:update', { detail: data });
        window.dispatchEvent(event);
    }
  };

  const fetchUser = async () => {
    const res = await fetch("/api/auth/me");
    if (res.ok) {
      const data = await res.json();
      setUserinfo(data);
    }
  };

  return (
    <header className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-8">
            <Link href="/" className="text-xl font-bold text-blue-600">
              ExperimentShop
            </Link>
            <nav className="flex space-x-4">
              <Link
                href="/products"
                className={`${
                  isAllProductsActive
                    ? "text-amber-600  "
                    : "text-gray-700 hover:text-blue-600"
                }`}
              >
                Products
              </Link>
              <Link
                href="/products?category=electronics"
                className={`${
                  isElectronicsActive
                    ? "text-amber-600 "
                    : "text-gray-700 hover:text-blue-600"
                }`}
              >
                Electronics
              </Link>
              <Link
                href="/products?category=home"
                className={`${
                  isHomeActive
                    ? "text-amber-600  "
                    : "text-gray-700 hover:text-blue-600"
                }`}
              >
                Home
              </Link>
              <Link
                href="/test-scenarios"
                className="text-purple-600 hover:text-purple-800 font-semibold"
              >
                Test Lab
              </Link>
            </nav>
          </div>

          <div className="flex items-center space-x-4">
            {sessionInfo && (
              <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                Group: {sessionInfo.experimentGroup}
              </span>
            )}
            <Link href="/cart" className="relative">
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z"
                />
              </svg>
              {cartCount > 0 && (
                <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                  {cartCount}
                </span>
              )}
            </Link>

            {userinfo ? (
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-600">
                  {userinfo.name} ({userinfo.role})
                </span>
                {userinfo.role === "admin" && (
                  <Link
                    href="/admin"
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    Admin
                  </Link>
                )}
                <button
                  onClick={handleLogout}
                  className="text-sm text-gray-500 hover:text-gray-700"
                >
                  Logout
                </button>
              </div>
            ) : (
              <Link
                href="/login"
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Login
              </Link>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

export default Navbar;
