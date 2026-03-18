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

  const [errors, setErrors] = useState<any>({
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

  // Sepeti çek
  useEffect(() => {
    fetch("/api/cart", { credentials: "include" })
      .then((res) => res.json())
      .then((data) => {
        setCart(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Helper Validasyonlar
  const isValidEmail = (email: string) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  
  const isValidPhone = (phone: string) =>
    /^\d{10,15}$/.test(phone); // sadece rakam, 10-15 hane
  
  const isValidName = (name: string) =>
    /^[a-zA-ZÇŞĞÜÖİçşğüöı\s]+$/.test(name.trim()); // Türkçe karakter + boşluk
  
  const isValidCardNumber = (num: string) => {
    const clean = num.replace(/\s|-/g, "");
    if (!/^\d{16}$/.test(clean)) return false;
    let sum = 0;
    for (let i = 0; i < 16; i++) {
      let digit = parseInt(clean[i]);
      if (i % 2 === 0) digit *= 2;
      if (digit > 9) digit -= 9;
      sum += digit;
    }
    return sum % 10 === 0;
  };

  const isValidExpiry = (exp: string) => {
    if (!/^\d{2}\/\d{2}$/.test(exp)) return false;
    const [mm, yy] = exp.split("/").map(Number);
    if (mm < 1 || mm > 12) return false;
    const now = new Date();
    const year = 2000 + yy;
    const month = mm - 1;
    const expDate = new Date(year, month + 1, 0);
    return expDate >= now;
  };

  const isValidCVV = (cvv: string) => /^\d{3,4}$/.test(cvv);

  const handleInputChange = (e: any) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));

    // Anlık hata kontrolü
    let error = "";
    switch (name) {
      case "name":
        if (!value.trim()) error = "Full name is required";
        else if (!isValidName(value)) error = "Invalid name";
        break;
      case "email":
        if (!value.trim()) error = "Email is required";
        else if (!isValidEmail(value)) error = "Invalid email";
        break;
      case "phone":
        if (!value.trim()) error = "Phone is required";
        else if (!isValidPhone(value)) error = "Invalid phone number";
        break;
      case "address":
        if (!value.trim()) error = "Address is required";
        break;
      case "city":
        if (!value.trim()) error = "City is required";
        break;
      case "cardNumber":
        if (!value.trim()) error = "Card number is required";
        else if (!isValidCardNumber(value)) error = "Invalid card number";
        break;
      case "cardName":
        if (!value.trim()) error = "Card name is required";
        else if (!isValidName(value)) error = "Invalid name";
        break;
      case "expiry":
        if (!value.trim()) error = "Expiry is required";
        else if (!isValidExpiry(value)) error = "Invalid expiry";
        break;
      case "cvv":
        if (!value.trim()) error = "CVV is required";
        else if (!isValidCVV(value)) error = "Invalid CVV";
        break;
    }
    setErrors((prev: any) => ({ ...prev, [name]: error }));
  };

  const validateStep1 = () => {
    const step1Fields = ["name", "email", "phone", "address", "city"];
    let valid = true;
    const newErrors: any = { ...errors };
    step1Fields.forEach((field) => {
      if (!formData[field].trim()) {
        newErrors[field] = "Required";
        valid = false;
      }
    });
    setErrors(newErrors);
    return valid;
  };

  const validateStep2 = () => {
    const step2Fields = ["cardNumber", "cardName", "expiry", "cvv"];
    let valid = true;
    const newErrors: any = { ...errors };
    step2Fields.forEach((field) => {
      if (!formData[field].trim()) {
        newErrors[field] = "Required";
        valid = false;
      }
    });
    setErrors(newErrors);
    return valid;
  };

  const processPayment = async () => {
    if (!validateStep1() || !validateStep2()) {
      alert("Please fix the errors before proceeding.");
      return;
    }

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

  if (loading) return <Layout><p className="text-center mt-20">Loading checkout...</p></Layout>;

  const items = Array.isArray(cart) ? cart : [];
  const subtotal = items.reduce((sum: number, item: any) => sum + item.price * item.quantity, 0);
  const tax = subtotal * 0.18;
  const shipping = subtotal > 500 ? 0 : 50;
  const total = subtotal + tax + shipping;

  return (
    <Layout>
      <div className="max-w-3xl mx-auto min-h-[calc(100vh-277px)]">
        <h1 className="text-2xl font-bold mb-8">Checkout</h1>

        {/* STEP 1 */}
        {step === 1 && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Shipping Information</h2>
            <div className="grid grid-cols-2 gap-4">
              <input name="name" placeholder="Full Name" className="border p-2 rounded" value={formData.name} onChange={handleInputChange} />
              {errors.name && <p className="text-red-600 text-sm col-span-2">{errors.name}</p>}

              <input name="email" placeholder="Email" className="border p-2 rounded" value={formData.email} onChange={handleInputChange} />
              {errors.email && <p className="text-red-600 text-sm col-span-2">{errors.email}</p>}

              <input name="phone" placeholder="Phone" className="border p-2 rounded" value={formData.phone} onChange={handleInputChange} />
              {errors.phone && <p className="text-red-600 text-sm col-span-2">{errors.phone}</p>}

              <input name="city" placeholder="City" className="border p-2 rounded" value={formData.city} onChange={handleInputChange} />
              {errors.city && <p className="text-red-600 text-sm col-span-2">{errors.city}</p>}

              <textarea name="address" placeholder="Full Address" rows={3} className="col-span-2 border p-2 rounded" value={formData.address} onChange={handleInputChange} />
              {errors.address && <p className="text-red-600 text-sm col-span-2">{errors.address}</p>}
            </div>
            <div className="mt-6 text-right">
              <button onClick={() => validateStep1() && setStep(2)} className="bg-blue-600 text-white px-6 py-3 rounded hover:bg-blue-700">Continue to Payment</button>
            </div>
          </div>
        )}

        {/* STEP 2 */}
        {step === 2 && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Payment Information</h2>
            

            <input name="cardNumber" placeholder="Card Number" className="border p-2 rounded w-full mb-1" value={formData.cardNumber} onChange={handleInputChange} />
            {errors.cardNumber && <p className="text-red-600 text-sm">{errors.cardNumber}</p>}

            <input name="cardName" placeholder="Name on Card" className="border p-2 rounded w-full mb-1" value={formData.cardName} onChange={handleInputChange} />
            {errors.cardName && <p className="text-red-600 text-sm">{errors.cardName}</p>}

            <div className="grid grid-cols-2 gap-4">
              <input name="expiry" placeholder="MM/YY" className="border p-2 rounded" value={formData.expiry} onChange={handleInputChange} />
              {errors.expiry && <p className="text-red-600 text-sm col-span-2">{errors.expiry}</p>}

              <input name="cvv" placeholder="CVV" className="border p-2 rounded" value={formData.cvv} onChange={handleInputChange} />
              {errors.cvv && <p className="text-red-600 text-sm col-span-2">{errors.cvv}</p>}
            </div>

            <div className="mt-6 flex justify-between">
              <button onClick={() => setStep(1)} className="border px-6 py-3 rounded hover:bg-gray-50">Back</button>
              <button onClick={() => validateStep2() && setStep(3)} className="bg-blue-600 text-white px-6 py-3 rounded hover:bg-blue-700">Review Order</button>
            </div>
          </div>
        )}

        {/* STEP 3 */}
        {step === 3 && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Confirm Order</h2>
            <p><strong>Shipping To:</strong><br />{formData.name}, {formData.address}, {formData.city}</p>
            <p className="mt-2"><strong>Payment:</strong> Card ending in {formData.cardNumber.slice(-4)}</p>

            <div className="mt-4 space-y-1">
              <p>Subtotal: ₺{subtotal.toFixed(2)}</p>
              <p>Tax (18%): ₺{tax.toFixed(2)}</p>
              <p>Shipping: ₺{shipping.toFixed(2)}</p>
              <p className="text-2xl font-bold text-blue-600">Total: ₺{total.toFixed(2)}</p>
            </div>

            <div className="mt-6 flex justify-between">
              <button onClick={() => setStep(2)} className="border px-6 py-3 rounded hover:bg-gray-50">Back</button>
              <button onClick={processPayment} className="bg-green-600 text-white px-8 py-3 rounded font-semibold hover:bg-green-700">Place Order</button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
