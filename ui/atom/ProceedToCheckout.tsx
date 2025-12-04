import { NextRouter, useRouter } from "next/router";
import React from "react";

function ProceedToCheckout() {
  const router: NextRouter = useRouter();
  return (
    <button
      onClick={() => router.push("/checkout")}
      className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 font-semibold"
    >
      Proceed to Checkout
    </button>
  );
}

export default ProceedToCheckout;
