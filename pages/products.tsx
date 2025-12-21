// pages/products.tsx
import Layout from "../components/Layout";
import { ChangeEvent, useEffect, useState } from "react";
import { NextRouter, useRouter } from "next/router";
import { products } from "../types/types";

export default function Products() {
  const [products, setProducts] = useState<products[]>([]);
  const [searchInput, setSearchInput] = useState<string>("");
  const [sortBy, setSortBy] = useState<string>("featured");
  const [loading, setLoading] = useState<boolean>(true);

  const router: NextRouter = useRouter();
  const { category, search } = router.query;

  useEffect(() => {
    fetchProducts();
  }, [category, search]);

  const fetchProducts = async () => {
    setLoading(true);
    const params = new URLSearchParams();
    if (category) params.append("category", category as string);
    if (search) params.append("search", search as string);

    const res = await fetch(`/api/products?${params}`);
    const data = await res.json();
    setProducts(data);
    setLoading(false);
  };

  // Ürünleri sıralama fonksiyonu
  const getSortedProducts = () => {
    const sorted = [...products];

    switch (sortBy) {
      case "price-low":
        return sorted.sort((a: products, b: products) => a.price - b.price);
      case "price-high":
        return sorted.sort((a: products, b: products) => b.price - a.price);
      case "featured":
      default:
        return sorted;
    }
  };

  const sortedProducts = getSortedProducts();

  // NOT: event (e) parametresi eklendi, böylece kart tıklamasını engelleyebiliriz.
  const addToCart = async (e: React.MouseEvent, productId: number, productTitle: string) => {
    // 1. ÖNEMLİ: Kartın tıklanma olayını durdur (Detay sayfasına gitmesin)
    e.stopPropagation();

    // 2. SDK Event'i (Mevcut kodun)
    const cartEvent = new CustomEvent("cart:add", {
      detail: {
        productId,
        productTitle,
        timestamp: Date.now(),
      },
    });
    window.dispatchEvent(cartEvent);
    console.log("[Products] cart:add event dispatched:", productTitle);

    // 3. API İsteği
    const res = await fetch("/api/cart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ productId }),
    });

    if (res.ok) {
      // 4. Navbar güncelleme sinyali
      window.dispatchEvent(new Event("cart:refresh"));
      console.log("Navbar'a güncelleme sinyali gönderildi.");
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      const params = new URLSearchParams();
      if (category) params.append("category", category as string);
      params.append("search", searchInput.trim());

      router.push(`/products?${params.toString()}`);
      setSearchInput("");
    } else {
      if (category) {
        router.push(`/products?category=${category}`);
      } else {
        router.push("/products");
      }
    }
  };

  return (
    <Layout>
      <div className="products-page flex gap-8 min-h-screen">
        {/* Filters Sidebar (Sol Taraf) */}
        <div className="w-64">
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="font-bold mb-4">Filters</h3>

            <div className="mb-6">
              <h4 className="font-semibold mb-2">Category</h4>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="category"
                    checked={!category}
                    onChange={() => router.push("/products")}
                  />
                  <span className="ml-2">All</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="category"
                    checked={category === "electronics"}
                    onChange={() => router.push("/products?category=electronics")}
                  />
                  <span className="ml-2">Electronics</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="category"
                    checked={category === "home"}
                    onChange={() => router.push("/products?category=home")}
                  />
                  <span className="ml-2">Home</span>
                </label>
              </div>
            </div>

            <form onSubmit={handleSearch}>
              <h4
                className={`font-bold mb-2 opacity-0 select-none ${
                  search?.length > 0 && "opacity-100"
                }`}
              >
                Filtered key: <span className="font-normal ">{search}</span>
              </h4>

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

        {/* Products Grid (Sağ Taraf) */}
        <div className="flex-1 ">
          <div className="mb-4 flex justify-between items-center ">
            <h1 className="text-2xl font-bold">
              {category
                ? `${
                    category.toString().charAt(0).toUpperCase() +
                    category.toString().slice(1)
                  } Products`
                : "All Products"}
            </h1>
            <select
              className="px-4 py-2 border rounded"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="featured">Sort by: Featured</option>
              <option value="price-low">Price: Low to High</option>
              <option value="price-high">Price: High to Low</option>
            </select>
          </div>

          {loading ? (
            <div className="grid grid-cols-3 gap-6 ">
              {[...Array(6)].map((_, i) => (
                <div
                  key={i}
                  className="bg-gray-200 animate-pulse rounded-lg h-64"
                ></div>
              ))}
            </div>
          ) : products.length === 0 ? (
            <h3 className="font-bold text-lg">No product found</h3>
          ) : (
            <div className="grid grid-cols-3 gap-6 ">
              {sortedProducts.map((product: any, index) => (
                <div
                  key={product.id}
                  data-order={index}
                  // YENİ: Karta tıklayınca detay sayfasına git
                  onClick={() => router.push(`/product/${product.id}`)}
                  className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow product-card cursor-pointer relative"
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
                        // YENİ: Event'i pass ediyoruz (e)
                        onClick={(e) => addToCart(e, product.id, product.title)}
                        className="
                          bg-gradient-to-r from-indigo-500 to-blue-600
                          hover:from-indigo-400 hover:to-blue-500
                          active:from-indigo-600 active:to-blue-700
                          text-white font-medium
                          px-3 py-1.5
                          rounded-lg
                          shadow-md hover:shadow-lg
                          transition-all duration-200
                        "
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