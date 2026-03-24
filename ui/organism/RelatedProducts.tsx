import React from "react";
import { products } from "../../types/types";

// ✅ Props tipi (TS hatasını çözer)
type Props = {
  currentProduct: products;
  products: products[];
};

function RelatedProducts({ currentProduct, products }: Props) {

  // 🔥 Related ürünleri bul
  const relatedProducts = (products || [])
    .filter(
      (p) =>
        p.category === currentProduct?.category &&
        p.id !== currentProduct?.id
    )
    .slice(0, 4); // max 4 ürün

  // ✅ boşsa hiçbir şey gösterme
  if (!relatedProducts.length) return null;

  return (
    <div className="mt-16">
      <h2 className="text-2xl font-bold mb-6">You May Also Like</h2>

      <div className="grid grid-cols-4 gap-6">
        {relatedProducts.map((product) => (
          <div
            key={product.id}
            className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow cursor-pointer"
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

              <p className="text-xl font-bold text-blue-600">
                ₺{product.price}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RelatedProducts;