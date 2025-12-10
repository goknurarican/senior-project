// pages/checkout.tsx
import Layout from '../components/Layout';
import { useState } from 'react';
import { useRouter } from 'next/router';

export default function Checkout() {
  const [step, setStep] = useState<number>(1);
  const [paymentAttempts, setPaymentAttempts] = useState<number>(0);
  const router = useRouter();
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    address: '',
    city: '',
    phone: '',
    cardNumber: '',
    cardName: '',
    expiry: '',
    cvv: ''
  });
  
  const handleInputChange = (e: any) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };
  
  const processPayment = async () => {
    setPaymentAttempts(paymentAttempts + 1);
    
    // Check if 3DS soft fail scenario is active
    if (window.ExperimentSDK?.firstPaymentAttempt) {
      window.ExperimentSDK.firstPaymentAttempt = false;
      alert('Payment failed. Please try again.');
      return;
    }
    
    // Simulate payment processing
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Success
    alert('Payment successful! Order #' + Math.random().toString(36).substr(2, 9).toUpperCase());
    router.push('/');
  };
  
  return (
    <Layout>
      <div className="max-w-3xl mx-auto min-h-[calc(100vh-277px)]">
        <h1 className="text-2xl font-bold mb-8">Checkout</h1>
        
        {/* Progress Steps */}
        <div className="flex mb-8">
          <div className={`flex-1 text-center pb-4 border-b-2 ${step >= 1 ? 'border-blue-600' : 'border-gray-300'}`}>
            <span className={`text-sm ${step >= 1 ? 'text-blue-600 font-semibold' : 'text-gray-500'}`}>
              1. Shipping Info
            </span>
          </div>
          <div className={`flex-1 text-center pb-4 border-b-2 ${step >= 2 ? 'border-blue-600' : 'border-gray-300'}`}>
            <span className={`text-sm ${step >= 2 ? 'text-blue-600 font-semibold' : 'text-gray-500'}`}>
              2. Payment
            </span>
          </div>
          <div className={`flex-1 text-center pb-4 border-b-2 ${step >= 3 ? 'border-blue-600' : 'border-gray-300'}`}>
            <span className={`text-sm ${step >= 3 ? 'text-blue-600 font-semibold' : 'text-gray-500'}`}>
              3. Confirm
            </span>
          </div>
        </div>
        
        {/* Step 1: Shipping */}
        {step === 1 && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Shipping Information</h2>
            <div className="grid grid-cols-2 gap-4">
              <input 
                name="name"
                placeholder="Full Name"
                className="px-3 py-2 border rounded"
                value={formData.name}
                onChange={handleInputChange}
              />
              <input 
                name="email"
                type="email"
                placeholder="Email"
                className="px-3 py-2 border rounded"
                value={formData.email}
                onChange={handleInputChange}
              />
              <input 
                name="phone"
                placeholder="Phone"
                className="px-3 py-2 border rounded"
                value={formData.phone}
                onChange={handleInputChange}
              />
              <input 
                name="city"
                placeholder="City"
                className="px-3 py-2 border rounded"
                value={formData.city}
                onChange={handleInputChange}
              />
              <textarea 
                name="address"
                placeholder="Full Address"
                className="col-span-2 px-3 py-2 border rounded"
                rows={3}
                value={formData.address}
                onChange={handleInputChange}
              />
            </div>
            
            <div className="mt-6 flex justify-end">
              <button 
                onClick={() => setStep(2)}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700">
                Continue to Payment
              </button>
            </div>
          </div>
        )}
        
        {/* Step 2: Payment */}
        {step === 2 && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Payment Information</h2>
            
            <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded">
              <p className="text-sm text-yellow-800">Test Mode: Use card number 4242 4242 4242 4242</p>
            </div>
            
            <div className="space-y-4">
              <input 
                name="cardNumber"
                placeholder="Card Number"
                className="w-full px-3 py-2 border rounded"
                value={formData.cardNumber}
                onChange={handleInputChange}
              />
              <input 
                name="cardName"
                placeholder="Name on Card"
                className="w-full px-3 py-2 border rounded"
                value={formData.cardName}
                onChange={handleInputChange}
              />
              <div className="grid grid-cols-2 gap-4">
                <input 
                  name="expiry"
                  placeholder="MM/YY"
                  className="px-3 py-2 border rounded"
                  value={formData.expiry}
                  onChange={handleInputChange}
                />
                <input 
                  name="cvv"
                  placeholder="CVV"
                  className="px-3 py-2 border rounded"
                  value={formData.cvv}
                  onChange={handleInputChange}
                />
              </div>
            </div>
            
            <div className="mt-6 flex justify-between">
              <button 
                onClick={() => setStep(1)}
                className="px-6 py-3 border rounded-lg hover:bg-gray-50">
                Back
              </button>
              <button 
                onClick={() => setStep(3)}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700">
                Review Order
              </button>
            </div>
          </div>
        )}
        
        {/* Step 3: Confirm */}
        {step === 3 && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Confirm Order</h2>
            
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold mb-2">Shipping To:</h3>
                <p className="text-gray-600">
                  {formData.name}<br />
                  {formData.address}<br />
                  {formData.city}<br />
                  {formData.phone}
                </p>
              </div>
              
              <div>
                <h3 className="font-semibold mb-2">Payment Method:</h3>
                <p className="text-gray-600">
                  Card ending in {formData.cardNumber.slice(-4)}
                </p>
              </div>
              
              <div>
                <h3 className="font-semibold mb-2">Order Total:</h3>
                <p className="text-2xl font-bold text-blue-600">₺2,567.89</p>
              </div>
            </div>
            
            <div className="mt-6 flex justify-between">
              <button 
                onClick={() => setStep(2)}
                className="px-6 py-3 border rounded-lg hover:bg-gray-50">
                Back
              </button>
              <button 
                onClick={processPayment}
                className="bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 font-semibold">
                Place Order
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
