// components/Layout.tsx
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

export default function Layout({ children }: { children: React.ReactNode }) {
  const [cartCount, setCartCount] = useState(0);
  const [sessionInfo, setSessionInfo] = useState<any>(null);
  const router = useRouter();
  
  useEffect(() => {
    fetchCartCount();
    fetchSessionInfo();
  }, [router.asPath]);
  
  const fetchCartCount = async () => {
    const res = await fetch('/api/cart');
    const items = await res.json();
    setCartCount(items.reduce((acc: number, item: any) => acc + item.quantity, 0));
  };
  
  const fetchSessionInfo = async () => {
    const res = await fetch('/api/session/info');
    const data = await res.json();
    setSessionInfo(data);
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-8">
              <Link href="/" className="text-xl font-bold text-blue-600">
                ExperimentShop
              </Link>
              <nav className="flex space-x-4">
                <Link href="/products" className="text-gray-700 hover:text-blue-600">
                  Products
                </Link>
                <Link href="/products?category=electronics" className="text-gray-700 hover:text-blue-600">
                  Electronics
                </Link>
                <Link href="/products?category=home" className="text-gray-700 hover:text-blue-600">
                  Home
                </Link>
              </nav>
            </div>
            
            <div className="flex items-center space-x-4">
              {sessionInfo && (
                <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                  Group: {sessionInfo.experimentGroup}
                </span>
              )}
              <Link href="/cart" className="relative">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
                {cartCount > 0 && (
                  <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                    {cartCount}
                  </span>
                )}
              </Link>
              <Link href="/admin" className="text-sm text-gray-500 hover:text-gray-700">
                Admin
              </Link>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
      
      {/* Footer */}
      <footer className="bg-gray-800 text-white mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-gray-400">
            © 2024 ExperimentShop - Research Platform
          </div>
        </div>
      </footer>
      
      {/* Load SDK */}
      <script src="/injection-sdk.js" defer></script>
    </div>
  );
}
