// pages/checkout.tsx
import Layout from "../components/Layout";
import { useEffect, useState } from "react";
import { useRouter } from "next/router";

export default function Checkout() {
  const [step, setStep] = useState(1);
  const [cart, setCart] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    address: "",
    city: "",
    phone: "",
    cardNumber: "",
    cardName: "",
    expiry: "",
    cvv: "",
  });

  // 🔥 CART'I SESSION İLE ÇEK
  useEffect(() => {
    fetch("/api/cart", {
      credentials: "include",
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("CHECKOUT CART:", data);
        setCart(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const items = Array.isArray(cart) ? cart : [];


  const subtotal = items.reduce(
    (sum: number, item: any) => sum + item.price * item.quantity,
    0
  );

  const tax = subtotal * 0.18;
  const shipping = subtotal > 500 ? 0 : 50;
  const total = subtotal + tax + shipping;

  const handleInputChange = (e: any) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const processPayment = async () => {
    // 3DS / experiment simülasyonu
    const sdk = (window as any).ExperimentSDK;

    if (sdk?.firstPaymentAttempt) {
      sdk.firstPaymentAttempt = false;
      alert("Payment failed. Please try again.");
      return;
    }

    await new Promise((r) => setTimeout(r, 1500));
    alert("Payment successful!");
    router.push("/");
  };

  if (loading) {
    return (
      <Layout>
        <p className="text-center mt-20">Loading checkout...</p>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="max-w-3xl mx-auto min-h-[calc(100vh-277px)]">
        <h1 className="text-2xl font-bold mb-8">Checkout</h1>

        {/* Progress */}
        <div className="flex mb-8">
          {["Shipping Info", "Payment", "Confirm"].map((label, i) => (
            <div
              key={label}
              className={`flex-1 text-center pb-4 border-b-2 ${
                step >= i + 1 ? "border-blue-600" : "border-gray-300"
              }`}
            >
              <span
                className={`text-sm ${
                  step >= i + 1
                    ? "text-blue-600 font-semibold"
                    : "text-gray-500"
                }`}
              >
                {i + 1}. {label}
              </span>
            </div>
          ))}
        </div>

        {/* STEP 1 */}
        {step === 1 && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">
              Shipping Information
            </h2>

            <div className="grid grid-cols-2 gap-4">
              <input name="name" placeholder="Full Name" className="border p-2 rounded" value={formData.name} onChange={handleInputChange} />
              <input name="email" placeholder="Email" className="border p-2 rounded" value={formData.email} onChange={handleInputChange} />
              <input name="phone" placeholder="Phone" className="border p-2 rounded" value={formData.phone} onChange={handleInputChange} />
              <input name="city" placeholder="City" className="border p-2 rounded" value={formData.city} onChange={handleInputChange} />
              <textarea name="address" placeholder="Address" rows={3} className="col-span-2 border p-2 rounded" value={formData.address} onChange={handleInputChange} />
            </div>

            <div className="mt-6 text-right">
              <button
                onClick={() => setStep(2)}
                className="bg-blue-600 text-white px-6 py-3 rounded"
              >
                Continue to Payment
              </button>
            </div>
          </div>
        )}

        {/* STEP 2 */}
        {step === 2 && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Payment</h2>

            <input name="cardNumber" placeholder="Card Number" className="border p-2 rounded w-full mb-3" value={formData.cardNumber} onChange={handleInputChange} />
            <input name="cardName" placeholder="Name on Card" className="border p-2 rounded w-full mb-3" value={formData.cardName} onChange={handleInputChange} />

            <div className="grid grid-cols-2 gap-4">
              <input name="expiry" placeholder="MM/YY" className="border p-2 rounded" value={formData.expiry} onChange={handleInputChange} />
              <input name="cvv" placeholder="CVV" className="border p-2 rounded" value={formData.cvv} onChange={handleInputChange} />
            </div>

            <div className="mt-6 flex justify-between">
              <button
                onClick={() => setStep(1)}
                className="border px-6 py-3 rounded"
              >
                Back
              </button>
              <button
                onClick={() => setStep(3)}
                className="bg-blue-600 text-white px-6 py-3 rounded"
              >
                Review Order
              </button>
            </div>
          </div>
        )}

        {/* STEP 3 */}
        {step === 3 && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Confirm Order</h2>

            <p>
              <strong>Shipping To:</strong>
              <br />
              {formData.name}, {formData.address}, {formData.city}
            </p>

            <p className="mt-2">
              <strong>Payment:</strong> Card ending in{" "}
              {formData.cardNumber.slice(-4)}
            </p>

            <div className="mt-4 space-y-1">
              <p>Subtotal: ₺{subtotal.toFixed(2)}</p>
              <p>Tax (18%): ₺{tax.toFixed(2)}</p>
              <p>Shipping: ₺{shipping.toFixed(2)}</p>
              <p className="text-2xl font-bold text-blue-600">
                Total: ₺{total.toFixed(2)}
              </p>
            </div>

            <div className="mt-6 flex justify-between">
              <button
                onClick={() => setStep(2)}
                className="border px-6 py-3 rounded"
              >
                Back
              </button>
              <button
                onClick={processPayment}
                className="bg-green-600 text-white px-8 py-3 rounded font-semibold"
              >
                Place Order
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
