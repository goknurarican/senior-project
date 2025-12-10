import "../styles/globals.css";
import type { AppProps } from "next/app";
import Script from "next/script";
import { useRouter } from "next/router";
import { useEffect } from "react";

export default function App({ Component, pageProps }: AppProps) {
  const router = useRouter();

  useEffect(() => {
    const handleRouteChange = (url: string) => {
      const adminAuthorize = () => {
        const isAdminUrl = url.startsWith("/admin");
        const userRole = document.cookie
          .split("; ")
          .find((row) => row.startsWith("user_permission="))
          ?.split("=")[1];

        if (isAdminUrl && userRole !== "admin") {
          alert("you are not authoried");
        }
      };
      adminAuthorize();
      const event = new CustomEvent("route:change", {
        detail: { url, timestamp: Date.now() },
      });
      window.dispatchEvent(event);
      console.log("[App] Route changed to:", url);
    };

    router.events.on("routeChangeComplete", handleRouteChange);

    return () => {
      router.events.off("routeChangeComplete", handleRouteChange);
    };
  }, [router.events]);

  return (
    <>
      <Script
        src="/injection-sdk.js"
        strategy="afterInteractive"
        onLoad={() => {
          console.log("[App] SDK loaded ✓");
          // Initialize the SDK after loading
          if (typeof window !== "undefined" && window.ExperimentSDK) {
            window.ExperimentSDK.init();
            console.log("[App] SDK initialized ✓");
          }
        }}
        onError={() => console.error("[App] SDK failed to load ✗")}
      />
      <Component {...pageProps} />
    </>
  );
}