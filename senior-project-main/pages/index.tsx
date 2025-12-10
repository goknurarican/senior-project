// pages/index.tsx
import Layout from "../components/Layout";
import { useEffect, useState } from "react";
import { products } from "../types/types";
import HeroSection from "../ui/molecule/HeroSection";
import CategoriesGrid from "../ui/organism/CategoriesGrid";
import FeaturedProduct from "../ui/organism/FeaturedProduct";
import Link from "next/link";

export default function Home() {
  const [featuredProducts, setFeaturedProducts] = useState<products[]>([]);

  useEffect(() => {
    fetch("/api/products")
      .then((res) => res.json())
      .then((data) => setFeaturedProducts(data.slice(0, 4)));
  }, []);
  useEffect(() => {});

  return (
    <Layout>
      {/* Hero Section */} {/* Artık componentini kullanmıyoruz. CSS gelmiyor. */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg p-12 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Welcome to ExperimentShop</h1>
        <p className="text-xl mb-8">
          Research Platform for E-commerce UX Testing
        </p>
        <Link
          href="/products"
          className="bg-white text-blue-600 px-6 py-3 rounded-lg font-medium hover:bg-gray-100"
        >
          Shop Now
        </Link>
      </div>

      {/* Categories Grid */}

      <div className="categories-grid mb-12">
        <h2 className="text-2xl font-bold mb-6">Categories</h2>
        <CategoriesGrid />
      </div>

      {/* Featured Products */}
      <div className="featured-products">
        <h2 className="text-2xl font-bold mb-6">Featured Products</h2>
        <div className="grid grid-cols-4 gap-6">
          {featuredProducts.map((product: any) => (
            <FeaturedProduct product={product} />
          ))}
        </div>
      </div>
    </Layout>
  );
}
