// pages/products.tsx
import Layout from "../components/Layout";
import { ChangeEvent, useEffect, useState } from "react";
import { NextRouter, useRouter } from "next/router";
import { products } from "../types/types";

export default function Products() {
  const [products, setProducts] = useState<products[]>([]);
  const [originalProducts, setOriginalProducts] = useState<products[]>([]);

  const [searchInput, setSearchInput] = useState<string>("");
  const [sortBy, setSortBy] = useState<string>("featured");
  const [loading, setLoading] = useState<boolean>(true);

  const router: NextRouter = useRouter();
  const { category, search } = router.query;

  /* --------------------------------------------------
      SORT RESET LISTENER (INJECTION -> UI)
  -------------------------------------------------- */
  useEffect(() => {
    const handleSortReset = () => {
      console.log("[Products] SORT RESET RECEIVED");
      setSortBy("featured");
    };

    window.addEventListener("sort:reset", handleSortReset);
    return () => {
      window.removeEventListener("sort:reset", handleSortReset);
    };
  }, []);
  

  //Face Reset için dinleyici (Filters -> UI)
  useEffect(() => {
  const handleFiltersReset = () => {
    setSearchInput('');       // Search input reset
    router.push("/products"); // Category All olarak reset
  };

  window.addEventListener("filters:reset", handleFiltersReset);
  return () => window.removeEventListener("filters:reset", handleFiltersReset);
  }, [router]);

   


  /* --------------------------------------------------
     📦 FETCH PRODUCTS
  -------------------------------------------------- */
  useEffect(() => {
    fetchProducts();
  }, [category, search]);

  const fetchProducts = async () => {
    setLoading(true);
    const params = new URLSearchParams();
    if (category) params.append("category", category as string);
    if (search) params.append("search", search as string);

    const res = await fetch(`/api/products?${params.toString()}`);
    const data = await res.json();
    setProducts(data);
    setOriginalProducts(data); // 🔥 SORT RESET İÇİN
    setLoading(false);
  };

  /* --------------------------------------------------
      SORT LOGIC
  -------------------------------------------------- */
  const getSortedProducts = () => {
  if (sortBy === "featured") {
    return [...originalProducts]; //  GERÇEK RESET mantğı burada
  }

  const sorted = [...products];
  switch (sortBy) {
    case "price-low":
      return sorted.sort((a, b) => a.price - b.price);
    case "price-high":
      return sorted.sort((a, b) => b.price - a.price);
    default:
      return sorted;
  }
};


  const sortedProducts = getSortedProducts();

  /* --------------------------------------------------
     🛒 ADD TO CART (Feedback + Injection uyumlu)
  -------------------------------------------------- */
  const addToCart = async (
    e: React.MouseEvent,
    productId: number,
    productTitle: string
  ) => {
    e.stopPropagation();

    const res = await fetch("/api/cart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ productId }),
    });

    if (res.ok) {
      // 🔔 Feedback (alert) – injection varsa gecikir
      alert("Product added to cart");

      // ➕ +1 Görsel Feedback
      const badge = document.createElement("div");
      badge.innerText = "+1";
      badge.style.cssText = `
        position: fixed;
        top: 80px;
        right: 40px;
        background: #2563eb;
        color: white;
        font-weight: bold;
        padding: 8px 12px;
        border-radius: 999px;
        z-index: 9999;
        animation: popFade 1.8s ease-out forwards;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.4);
      `;

      const style = document.createElement("style");
      style.innerHTML = `
        @keyframes popFade {
          0%   { transform: scale(0.5); opacity: 0; }
          20%  { transform: scale(1.2); opacity: 1; }
          80%  { transform: scale(1); opacity: 1; }
          100% { transform: scale(0.8); opacity: 0; }
        }
      `;

      document.head.appendChild(style);
      document.body.appendChild(badge);

      setTimeout(() => {
        badge.remove();
        style.remove();
      }, 1800);

      // 🛒 Navbar update
      window.dispatchEvent(new Event("cart:refresh"));
    }
  };

  /* --------------------------------------------------
     🔍 SEARCH
  -------------------------------------------------- */
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      const params = new URLSearchParams();
      if (category) params.append("category", category as string);
      params.append("search", searchInput.trim());
      router.push(`/products?${params.toString()}`);
      setSearchInput("");
    } else {
      category
        ? router.push(`/products?category=${category}`)
        : router.push("/products");
    }
  };

  /* --------------------------------------------------
     🖼️ UI geliştirmeli 
  -------------------------------------------------- */
  return (
    <Layout>
      <div className="products-page flex gap-8 min-h-screen">
        {/* Filters */}
        <div className="w-64">
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="font-bold mb-4">Filters</h3>

            <div className="mb-6">
              <h4 className="font-semibold mb-2">Category</h4>
              <div className="space-y-2">
                {["all", "electronics", "home"].map((cat) => (
                  <label key={cat} className="flex items-center">
                    <input
                      type="radio"
                      checked={category ? category === cat : cat === "all"}
                      onChange={() =>
                        cat === "all"
                          ? router.push("/products")
                          : router.push(`/products?category=${cat}`)
                      }
                    />
                    <span className="ml-2">
                      {cat.charAt(0).toUpperCase() + cat.slice(1)}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <form onSubmit={handleSearch}>
              <h4 className="font-semibold mb-2">Search</h4>
              <input
                type="text"
                placeholder="Search products..."
                className="w-full px-3 py-2 border rounded"
                value={searchInput}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  setSearchInput(e.target.value)
                }
              />
            </form>
          </div>
        </div>

        {/* Products */}
        <div className="flex-1">
          <div className="mb-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold">
              {category
                ? `${category.toString().toUpperCase()} Products`
                : "All Products"}
            </h1>

            {/*  KEY EKLENDİ → SORT RESET %100 çalışıyor */}
            <select
              key={sortBy}
              className="px-4 py-2 border rounded"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="featured">Featured</option>
              <option value="price-low">Price: Low to High</option>
              <option value="price-high">Price: High to Low</option>
            </select>
          </div>

          {loading ? (
            <div className="grid grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <div
                  key={i}
                  className="bg-gray-200 animate-pulse rounded-lg h-64"
                />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-6">
              {sortedProducts.map((product) => (
                <div
                  key={product.id}
                  onClick={() => router.push(`/product/${product.id}`)}
                  className="product-card bg-white rounded-lg shadow hover:shadow-lg transition-shadow cursor-pointer"
                >
                  <img
                    src={product.image}
                    alt={product.title}
                    className="w-full h-48 object-cover rounded-t-lg product-image"
                  />
                  <div className="p-4">
                    <h3 className="font-semibold mb-2">{product.title}</h3>
                    <p className="text-gray-600 text-sm mb-3">
                      {product.description?.substring(0, 60)}...
                    </p>
                    <div className="flex justify-between items-center">
                      <span className="text-xl font-bold text-blue-600">
                        ₺{product.price}
                      </span>
                      <button
                        onClick={(e) =>
                          addToCart(e, product.id, product.title)
                        }
                        className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg"
                      >
                        Add to Cart
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
