import Layout from "../components/Layout";
import { ChangeEvent, useEffect, useState } from "react";
import { useRouter } from "next/router";
import { products as ProductType } from "../types/types";

export default function Products() {
  const [products, setProducts] = useState<ProductType[]>([]);
  const [searchInput, setSearchInput] = useState("");
  const [sortBy, setSortBy] = useState("featured");
  const [loading, setLoading] = useState(true);

  const router = useRouter();
  const { category, search } = router.query;

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
    setLoading(false);
  };

  const getSortedProducts = () => {
    const list = [...products];
    if (sortBy === "price-low") return list.sort((a, b) => a.price - b.price);
    if (sortBy === "price-high") return list.sort((a, b) => b.price - a.price);
    return list;
  };

  const addToCart = async (productId: number, productTitle: string) => {
    await fetch("/api/cart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ productId }),
    });

    alert(`${productTitle} sepete eklendi`);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();

    if (searchInput.trim()) {
      const params = new URLSearchParams();
      if (category) params.append("category", category as string);
      params.append("search", searchInput.trim());
      router.push(`/products?${params.toString()}`);
    } else {
      router.push("/products");
    }
  };

  return (
    <Layout>
      <div className="products-page flex gap-8 min-h-screen">
        {/* Filters */}
        <div className="w-64">
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="font-bold mb-4">Filters</h3>

            <form onSubmit={handleSearch}>
              <input
                value={searchInput}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  setSearchInput(e.target.value)
                }
                placeholder="Search products..."
                className="w-full border px-3 py-2 rounded"
              />
            </form>
          </div>
        </div>

        {/* Products */}
        <div className="flex-1">
          <div className="mb-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold">
              {category ? `${category} Products` : "All Products"}
            </h1>

            <select
              className="border px-3 py-2 rounded"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="featured">Featured</option>
              <option value="price-low">Price: Low to High</option>
              <option value="price-high">Price: High to Low</option>
            </select>
          </div>

          {loading ? (
            <p>Loading...</p>
          ) : (
            <div className="grid grid-cols-3 gap-6">
              {getSortedProducts().map((product) => (
                <div
                  key={product.id}
                  onClick={() =>
                    (window.location.href = `/product/${product.id}`)
                  }
                  style={{
                    pointerEvents: "auto",
                    zIndex: 10,
                  }}
                  className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow cursor-pointer relative"
                >
                  <img
                    src={product.image}
                    alt={product.title}
                    className="w-full h-48 object-cover rounded-t-lg"
                  />

                  <div className="p-4">
                    <h3 className="font-semibold mb-2">
                      {product.title}
                    </h3>

                    <p className="text-gray-600 text-sm mb-3">
                      {product.description?.substring(0, 60)}...
                    </p>

                    <div className="flex justify-between items-center">
                      <span className="text-xl font-bold text-blue-600">
                        ₺{product.price}
                      </span>

                      <button
                        onClick={(e) => {
                          e.stopPropagation(); // karta gitmesini engeller
                          addToCart(product.id, product.title);
                        }}
                        className="bg-blue-600 text-white px-3 py-1.5 rounded"
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
