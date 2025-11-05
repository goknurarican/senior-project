// pages/index.tsx
import Layout from '../components/Layout';
import Link from 'next/link';
import { useEffect, useState } from 'react';

export default function Home() {
  const [featuredProducts, setFeaturedProducts] = useState([]);
  
  useEffect(() => {
    fetch('/api/products')
      .then(res => res.json())
      .then(data => setFeaturedProducts(data.slice(0, 4)));
  }, []);
  
  return (
    <Layout>
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg p-12 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Welcome to ExperimentShop</h1>
        <p className="text-xl mb-8">Research Platform for E-commerce UX Testing</p>
        <Link href="/products" className="bg-white text-blue-600 px-6 py-3 rounded-lg font-medium hover:bg-gray-100">
          Shop Now
        </Link>
      </div>
      
      {/* Categories Grid */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Categories</h2>
        <div className="grid grid-cols-2 gap-6">
          <Link href="/products?category=electronics" 
            className="bg-white p-8 rounded-lg shadow hover:shadow-lg transition-shadow">
            <h3 className="text-xl font-semibold mb-2">Electronics</h3>
            <p className="text-gray-600">Latest tech gadgets and accessories</p>
          </Link>
          <Link href="/products?category=home" 
            className="bg-white p-8 rounded-lg shadow hover:shadow-lg transition-shadow">
            <h3 className="text-xl font-semibold mb-2">Home & Office</h3>
            <p className="text-gray-600">Furniture and home essentials</p>
          </Link>
        </div>
      </div>
      
      {/* Featured Products */}
      <div>
        <h2 className="text-2xl font-bold mb-6">Featured Products</h2>
        <div className="grid grid-cols-4 gap-6">
          {featuredProducts.map((product: any) => (
            <div key={product.id} className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow product-card">
              <img 
                src={product.image} 
                alt={product.title}
                className="w-full h-48 object-cover rounded-t-lg product-image"
              />
              <div className="p-4">
                <h3 className="font-semibold mb-2">{product.title}</h3>
                <p className="text-xl font-bold text-blue-600">₺{product.price}</p>
                <Link href={`/product/${product.id}`} 
                  className="mt-3 block text-center bg-blue-600 text-white py-2 rounded hover:bg-blue-700">
                  View Details
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Layout>
  );
}
