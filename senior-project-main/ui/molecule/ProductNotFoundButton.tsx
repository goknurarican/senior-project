import Link from "next/link";
import React from "react";

function ProductNotFoundPage() {
  return (
    <div className="text-center py-16 bg-black">
      <h1 className="text-2xl font-bold mb-4">Product Not Found</h1>
      <Link href="/products" className="text-blue-600 hover:underline">
        Back to Products
      </Link>
    </div>
  );
}

export default ProductNotFoundPage;
