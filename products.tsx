// pages/products.tsx
import Layout from '../components/Layout';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';

export default function Products() {
  const [products, setProducts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const { category, search, sort: sortQuery } = router.query;

  // Keep sort state in sync with URL
  const [sort, setSort] = useState<string>(sortQuery ? (sortQuery as string) : 'featured');

  useEffect(() => {
    fetchProducts();
  }, [category, search]);

  useEffect(() => {
    // Update URL when sort changes
    const params = new URLSearchParams();
    if (category) params.append('category', category as string);
    if (search) params.append('search', search as string);
    if (sort && sort !== 'featured') params.append('sort', sort);

    router.replace(`/products?${params.toString()}`, undefined, { shallow: true });

    // Sort products locally
    let sortedProducts = [...products];
    if (sort === 'lowToHigh') sortedProducts.sort((a, b) => a.price - b.price);
    if (sort === 'highToLow') sortedProducts.sort((a, b) => b.price - a.price);
    setProducts(sortedProducts);
  }, [sort]);

  const fetchProducts = async () => {
    setLoading(true);
    const params = new URLSearchParams();
    if (category) params.append('category', category as string);
    if (search) params.append('search', search as string);

    const res = await fetch(`/api/products?${params.toString()}`);
    const data = await res.json();
    setProducts(data);
    setLoading(false);
  };

  const addToCart = async (productId: number) => {
    await fetch('/api/cart', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ productId })
    });

    // Refresh cart count
    window.location.reload();
  };

  return (
    <Layout>
      <div className="flex gap-8">
        {/* Filters */}
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
                    onChange={() => router.push('/products')}
                  />
                  <span className="ml-2">All</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="category"
                    checked={category === 'electronics'}
                    onChange={() => router.push('/products?category=electronics')}
                  />
                  <span className="ml-2">Electronics</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="category"
                    checked={category === 'home'}
                    onChange={() => router.push('/products?category=home')}
                  />
                  <span className="ml-2">Home</span>
                </label>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Search</h4>
              <input
                type="text"
                placeholder="Search products..."
                className="w-full px-3 py-2 border rounded"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    router.push(
                      `/products?search=${(e.target as HTMLInputElement).value}${
                        category ? `&category=${category}` : ''
                      }${sort !== 'featured' ? `&sort=${sort}` : ''}`
                    );
                  }
                }}
              />
            </div>
          </div>
        </div>

        {/* Products Grid */}
        <div className="flex-1">
          <div className="mb-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold">
              {category
                ? `${category.toString().charAt(0).toUpperCase() + category.toString().slice(1)} Products`
                : 'All Products'}
            </h1>
            <select
              className="px-4 py-2 border rounded"
              value={sort}
              onChange={(e) => setSort(e.target.value)}
            >
              <option value="featured">Sort by: Featured</option>
              <option value="lowToHigh">Price: Low to High</option>
              <option value="highToLow">Price: High to Low</option>
            </select>
          </div>

          {loading ? (
            <div className="grid grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="bg-gray-200 animate-pulse rounded-lg h-64"></div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-6">
              {products.map((product, index) => (
                <div
                  key={product.id}
                  data-order={index}
                  className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow product-card"
                >
                  <img
                    src={product.image}
                    alt={product.title}
                    className="w-full h-48 object-cover rounded-t-lg product-image"
                  />
                  <div className="p-4">
                    <h3 className="font-semibold mb-2">{product.title}</h3>
                    <p className="text-gray-600 text-sm mb-3">{product.description?.substring(0, 60)}...</p>
                    <div className="flex justify-between items-center">
                      <span className="text-xl font-bold text-blue-600">₺{product.price}</span>
                      <button
                        onClick={() => addToCart(product.id)}
                        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 add-to-cart"
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


