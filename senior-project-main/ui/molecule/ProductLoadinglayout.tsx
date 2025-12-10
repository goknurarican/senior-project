import React from "react";

function ProductLoadinglayout() {
  return (
    <div className="animate-pulse">
      <div className="grid grid-cols-2 gap-8">
        <div className="h-96 bg-gray-200 rounded"></div>
        <div className="space-y-4">
          <div className="h-8 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          <div className="h-24 bg-gray-200 rounded"></div>
        </div>
      </div>
    </div>
  );
}
export default ProductLoadinglayout;