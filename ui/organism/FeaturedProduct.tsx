import Link from "next/link";
import React from "react";
import { products } from "../../types/types";

type FeaturedProductProps = {
  product: products;
};

function FeaturedProduct({ product }: FeaturedProductProps) {
  return (
    // DIŞTAKİ LİNK KALIYOR (Tüm kart tıklanabilir)
    <Link
      href={`/product/${product.id}`}
      key={product.id}
      className="product-card bg-white rounded-lg shadow hover:shadow-lg transition-shadow block"
    >
      <img
        src={product.image}
        alt={product.title}
        className="w-full h-48 object-cover rounded-t-lg product-image"
      />
      <div className="p-4">
        <h3 className="font-semibold mb-2">{product.title}</h3>
        <p className="text-xl font-bold text-blue-600">₺{product.price}</p>

        {/* İÇTEKİ LİNK span'a ÇEVRİLDİ (Hata buradaydı) */}
        <span
          className="mt-3 block text-center bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
        >
          View Details
        </span>
      </div>
    </Link>
  );
}

export default FeaturedProduct;