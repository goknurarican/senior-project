import React from "react";

function RelatedProducts() {
  return (
    <div className="mt-16">
      <h2 className="text-2xl font-bold mb-6">You May Also Like</h2>
      <div className="grid grid-cols-4 gap-6">
        {/* Mock related products */}
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow"
          >
            <img
              src={`https://picsum.photos/400/300?random=${i + 20}`}
              alt="Related product"
              className="w-full h-48 object-cover rounded-t-lg"
            />
            <div className="p-4">
              <h3 className="font-semibold mb-2">Related Product {i}</h3>
              <p className="text-xl font-bold text-blue-600">
                ₺{(999 + i * 100).toFixed(2)}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RelatedProducts;
