// pages/cart.tsx
import Layout from "../components/Layout";
import { useEffect, useState } from "react";
import { useCartStore } from "../storage/zustand";
import CartLoadingLayout from "../ui/molecule/CartLoadingLayout";
import EmptyCartLayout from "../ui/molecule/EmptyCartLayout";
import CartItems from "../ui/organism/CartItem";
import { CartItem } from "../types/types";
import SubtotalArea from "../ui/organism/SubtotalArea";

export default function Cart() {
  const cartCount = useCartStore((state) => state.cartItem);

  const setCartCount = useCartStore((state) => state.setCartItem);

  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchCart();
  }, []);

  const fetchCart = async () => {
    const res = await fetch("/api/cart");
    const data = await res.json();
    console.log("cart items : ", data);
    setCartItems(data);
    setLoading(false);
  };

  const removeItem = async (productId: number) => {
    const item = cartItems.find((item: any) => item.product_id === productId);
    if (item) {
      setCartCount(cartCount - item.quantity);
    }

    // 3. API'dan silme işlemini yap
    await fetch(`/api/cart?productId=${productId}`, {
      method: "DELETE",
    });

    // 4. Listeyi yenile
    fetchCart();
  };

  const updateQuantity = async (productId: number, change: number) => {
    const item = cartItems.find((i: any) => i.product_id === productId);
    setCartCount(cartCount + change);
    if (change < 0 && item && item.quantity <= 1) {
      // Quantity 1'e düşünce direkt siliyoruz.
      await removeItem(productId);
      return;
    }

    // Quantity'yi güncelle
    await fetch("/api/cart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        productId,
        quantity: change, // veya quantity: item.quantity + change
      }),
    });

    fetchCart();
  };

  const calculateTotal = () => {
    return cartItems.reduce(
      (acc: number, item: any) => acc + item.price * item.quantity,
      0
    );
  };

  if (loading) {
    return (
      <Layout>
        <CartLoadingLayout />
      </Layout>
    );
  }

  if (cartItems.length === 0) {
    return (
      <Layout>
        <EmptyCartLayout />
      </Layout>
    );
  }

  return (
    <Layout>
      <h1 className="text-2xl font-bold mb-8">Shopping Cart</h1>

      <div className="grid grid-cols-3 gap-8 ">
        <div className="col-span-2">
          <div className="bg-white rounded-lg shadow p-6">
            {cartItems.map((item: any) => (
              <div
                key={item.id}
                className="flex items-center gap-4 py-4 border-b last:border-0"
              >
                <img
                  src={item.image}
                  alt={item.title}
                  className="w-24 h-24 object-cover rounded"
                />
                <div className="flex-1">
                  <h3 className="font-semibold">{item.title}</h3>
                  <p className="text-gray-600">₺{item.price}</p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => updateQuantity(item.product_id, -1)}
                    className="w-8 h-8 rounded border hover:bg-gray-100"
                  >
                    -
                  </button>
                  <span className="w-12 text-center">{item.quantity}</span>
                  <button
                    onClick={() => updateQuantity(item.product_id, 1)}
                    className="w-8 h-8 rounded border hover:bg-gray-100"
                  >
                    +
                  </button>
                </div>
                <div className="text-lg font-semibold">
                  ₺{(item.price * item.quantity).toFixed(2)}
                </div>
                <button
                  onClick={() => removeItem(item.product_id)}
                  className="text-red-500 hover:text-red-700"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* <div> */}
        <SubtotalArea calculateTotal={calculateTotal} />
        {/* </div> */}
      </div>
    </Layout>
  );
}
