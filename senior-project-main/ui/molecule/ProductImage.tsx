import React, { Dispatch, SetStateAction } from "react";
import { products } from "../../types/types";
type ProductImageProps = {
  images: string[];
  selectedImage: number;
  setSelectedImage: React.Dispatch<React.SetStateAction<number>>;
  product: products;
};
function ProductImage({
  images,
  selectedImage,
  setSelectedImage,
  product,
}: ProductImageProps) {
  return (
    <div>
      <div className="mb-4">
        <img
          src={images[selectedImage]}
          alt={product.title}
          className="w-full h-96 object-cover rounded-lg product-image"
        />
      </div>
      <div className="flex gap-2">
        {images.map((img, index) => (
          <button
            key={index}
            onClick={() => setSelectedImage(index)}
            className={`border-2 rounded ${
              selectedImage === index ? "border-blue-500" : "border-gray-300"
            }`}
          >
            <img
              src={img}
              alt={`${product.title} ${index + 1}`}
              className="w-20 h-20 object-cover rounded"
            />
          </button>
        ))}
      </div>
    </div>
  );
}

export default ProductImage;
