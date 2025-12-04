import React from "react";

function TestProduct() {
  return (
    <div className="bg-white rounded-lg shadow p-4 product-card">
      <img
        src="https://picsum.photos/400/300?random=test"
        alt="Test Product"
        className="w-full h-48 object-cover rounded product-image mb-4"
      />
      <h3 className="font-semibold mb-2">Test Product</h3>
      <p className="text-gray-600 mb-3">
        This is a test product for scenario testing
      </p>
      <button className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 add-to-cart">
        Add to Cart
      </button>
    </div>
  );
}

export default TestProduct;
