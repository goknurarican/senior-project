import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(req: NextRequest ) {
  const { pathname } = req.nextUrl;

  // Sadece /admin yollarını korumak istiyoruz. Oğuzhan.
  const isAdminRoute = pathname.startsWith("/admin");

  if (isAdminRoute) {
    const role = req.cookies.get("user_permission")?.value;

    if (role !== "admin") {
      const url = req.nextUrl.clone();
      url.pathname = "/unauthorized"; // Yönlendirilecek sayfa
      return NextResponse.redirect(url);
    }
  }

  return NextResponse.next();
}
export const config = {
  matcher: ["/admin/:path*"],
};