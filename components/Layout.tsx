// components/Layout.tsx
import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import Footer from "./Footer";
import Navbar from "./Navbar";
import ExperimentBar from '../ui/organism/ExperimentBar';

export default function Layout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [isAuthorized, setIsAuthorized] = useState(false);

  useEffect(() => {
    // Kullanıcı zaten login veya signup sayfasındaysa geçişe izin ver
    if (router.pathname === '/login' || router.pathname === '/signup') {
      setIsAuthorized(true);
      return;
    }

    // Diğer tüm sayfalarda kullanıcının kayıtlı olup olmadığını kontrol et
    fetch('/api/auth/me')
      .then(res => {
        if (res.status === 401) {
          // Giriş yapmamışsa (anonimse) zorla kayıt sayfasına yönlendir
          router.push('/signup');
        } else {
          // Kayıtlıysa sitenin açılmasına izin ver
          setIsAuthorized(true);
        }
      })
      .catch(() => setIsAuthorized(true));
  }, [router.pathname]);

  // Kontrol sürerken ana sayfanın saliselik de olsa görünmesini engelle (Beyaz ekran)
  if (!isAuthorized) return null;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <Navbar />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <Footer />

      {/* Deney Kontrol Çubuğu */}
      <ExperimentBar />
    </div>
  );
}