import Link from "next/link";
import React from "react";

function HeroSection() {
  return (
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
  );
}

export default HeroSection;
