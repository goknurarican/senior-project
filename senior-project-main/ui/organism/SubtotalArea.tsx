import React from "react";
import ProceedToCheckout from "../atom/ProceedToCheckout";
import CouponCode from "../atom/CouponCode";

function SubtotalArea({ calculateTotal }: { calculateTotal: () => number }) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-4">Order Summary</h2>

      <div className="space-y-2 mb-4">
        <div className="flex justify-between">
          <span>Subtotal</span>
          <span>₺{calculateTotal().toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span>Tax (18%)</span>
          <span>₺{(calculateTotal() * 0.18).toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span>Shipping</span>
          <span>₺50.00</span>
        </div>
      </div>

      <div className="border-t pt-4 mb-6">
        <div className="flex justify-between font-bold text-lg">
          <span>Total</span>
          <span>₺{(calculateTotal() * 1.18 + 50).toFixed(2)}</span>
        </div>
      </div>

      <CouponCode />

      <ProceedToCheckout />
    </div>
  );
}

export default SubtotalArea;
