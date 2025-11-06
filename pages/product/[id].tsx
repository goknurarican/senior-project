// pages/product/[id].tsx
import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';
import Layout from '../../components/Layout';
import Link from 'next/link';

export default function ProductDetail() {
  const router = useRouter();
  const { id } = router.query;
  const [product, setProduct] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [quantity, setQuantity] = useState(1);
  const [selectedImage, setSelectedImage] = useState(0);

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
    await fetch('/api/cart', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ productId: product.id, quantity })
    });
    
    // Show success message
    alert(`Added ${quantity} x ${product.title} to cart!`);
    router.push('/cart');
  };

  if (loading) {
    return (
      <Layout>
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
      </Layout>
    );
  }

  if (!product) {
    return (
      <Layout>
        <div className="text-center py-16">
          <h1 className="text-2xl font-bold mb-4">Product Not Found</h1>
          <Link href="/products" className="text-blue-600 hover:underline">
            Back to Products
          </Link>
        </div>
      </Layout>
    );
  }

  // Generate multiple images for gallery
  const images = [
    product.image,
    product.image.replace('random=', 'random=a'),
    product.image.replace('random=', 'random=b'),
    product.image.replace('random=', 'random=c'),
  ];

  return (
    <Layout>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Product Images */}
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
                className={`border-2 rounded ${selectedImage === index ? 'border-blue-500' : 'border-gray-300'}`}
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

        {/* Product Info */}
        <div>
          <nav className="text-sm mb-4">
            <Link href="/" className="text-gray-500 hover:text-gray-700">Home</Link>
            <span className="mx-2">/</span>
            <Link href="/products" className="text-gray-500 hover:text-gray-700">Products</Link>
            <span className="mx-2">/</span>
            <Link href={`/products?category=${product.category}`} className="text-gray-500 hover:text-gray-700">
              {product.category}
            </Link>
            <span className="mx-2">/</span>
            <span className="text-gray-900">{product.title}</span>
          </nav>

          <h1 className="text-3xl font-bold mb-4">{product.title}</h1>
          
          <div className="mb-6">
            <span className="text-3xl font-bold text-blue-600">₺{product.price}</span>
            <span className="ml-3 text-sm text-gray-500">VAT Included</span>
          </div>

          <div className="mb-6">
            <p className="text-gray-700">{product.description}</p>
          </div>

          <div className="mb-6">
            <h3 className="font-semibold mb-2">Features:</h3>
            <ul className="list-disc list-inside text-gray-600 space-y-1">
              <li>High quality materials</li>
              <li>2 year warranty included</li>
              <li>Free shipping over ₺500</li>
              <li>30-day return policy</li>
            </ul>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quantity
            </label>
            <div className="flex items-center gap-2">
              <button 
                onClick={() => setQuantity(Math.max(1, quantity - 1))}
                className="w-10 h-10 rounded border hover:bg-gray-100">
                -
              </button>
              <input 
                type="number" 
                value={quantity}
                onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 1))}
                className="w-20 text-center border rounded px-2 py-1"
              />
              <button 
                onClick={() => setQuantity(quantity + 1)}
                className="w-10 h-10 rounded border hover:bg-gray-100">
                +
              </button>
            </div>
          </div>

          <div className="flex gap-4">
            <button 
              onClick={addToCart}
              className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 font-semibold add-to-cart">
              Add to Cart
            </button>
            <button className="px-6 py-3 border border-gray-300 rounded-lg hover:bg-gray-50">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
              </svg>
            </button>
          </div>

          <div className="mt-8 border-t pt-8">
            <div className="space-y-2 text-sm text-gray-600">
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                In Stock ({product.stock || 100} available)
              </div>
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Ships in 1-3 business days
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Related Products */}
      <div className="mt-16">
        <h2 className="text-2xl font-bold mb-6">You May Also Like</h2>
        <div className="grid grid-cols-4 gap-6">
          {/* Mock related products */}
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow">
              <img 
                src={`https://picsum.photos/400/300?random=${i + 20}`} 
                alt="Related product"
                className="w-full h-48 object-cover rounded-t-lg"
              />
              <div className="p-4">
                <h3 className="font-semibold mb-2">Related Product {i}</h3>
                <p className="text-xl font-bold text-blue-600">₺{(999 + i * 100).toFixed(2)}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Layout>
  );
}
