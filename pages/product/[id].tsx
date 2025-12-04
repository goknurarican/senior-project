// pages/product/[id].tsx
import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import Layout from "../../components/Layout";
import { products } from "../../types/types";
import { getCookie } from "cookies-next";
import ProductNotFoundPage from "../../ui/molecule/ProductNotFoundButton";
import ProductLoadinglayout from "../../ui/molecule/ProductLoadinglayout";
import ProductImage from "../../ui/molecule/ProductImage";
import ProductInfo from "../../ui/organism/ProductInfo";
import RelatedProducts from "../../ui/organism/RelatedProducts";

export default function ProductDetail() {
  const router = useRouter();
  const { id } = router.query;
  const [product, setProduct] = useState<products>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [quantity, setQuantity] = useState<number>(1);
  const [selectedImage, setSelectedImage] = useState<number>(0);

  useEffect(() => {
    if (id) {
      fetchProduct();
    }
  }, [id]);

  const fetchProduct = async () => {
    const res = await fetch(`/api/products/${id}`);
    const data = await res.json();
    setProduct(data);
    setLoading(false);
  };

  const addToCart = async () => {
    await fetch("/api/cart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ productId: product.id, quantity }),
    });

    // Show success message
    alert(`Added ${quantity} x ${product.title} to cart!`);
    router.push("/cart");
  };

  if (loading) {
    return (
      <Layout>
        <ProductLoadinglayout />
      </Layout>
    );
  }

  if (!product) {
    return (
      <Layout>
        <ProductNotFoundPage />
      </Layout>
    );
  }

  // Generate multiple images for gallery
  const images = [
    product.image,
    product.image.replace("random=", "random=a"),
    product.image.replace("random=", "random=b"),
    product.image.replace("random=", "random=c"),
  ];

  return (
    <Layout>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Product Images */}
        <ProductImage
          images={images}
          selectedImage={selectedImage}
          setSelectedImage={setSelectedImage}
          product={product}
        />
        {/* Product Info */}
        <ProductInfo
          product={product}
          quantity={quantity}
          setQuantity={setQuantity}
          addToCart={addToCart}
        />
      </div>
      {/* Related Products */}
      <RelatedProducts />
    </Layout>
  );
}
