import Link from "next/link";
import React from "react";

function EmptyCartLayout() {
  return (
    <div className="text-center py-16 min-h-[calc(100vh-277px)]">
      <h1 className="text-2xl font-bold mb-4">Your Cart is Empty</h1>
      <p className="text-gray-600 mb-8">Add some products to get started!</p>
      <Link
        href="/products"
        className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700"
      >
        Continue Shopping
      </Link>
    </div>
  );
}

export default EmptyCartLayout;
