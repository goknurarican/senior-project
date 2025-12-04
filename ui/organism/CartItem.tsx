import React from "react";
import { CartItem } from "../../types/types";
type CartItemProps = {
  item: CartItem;
  updateQuantity: (productId: number, change: number) => Promise<void>;
  removeItem: (productId: number) => Promise<void>;
};
function CartItems({ item, updateQuantity, removeItem }: CartItemProps) {
  return (
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
  );
}

export default CartItems;
