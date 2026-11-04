import type { Metadata, Viewport } from "next";
import Link from "next/link";
import UserBadge from "@/components/UserBadge";
import "./globals.css";

export const metadata: Metadata = {
  title: "WREN Story Engine",
  description: "Deterministic multi-agent pipeline for consistent multi-scene stories",
};
export const viewport: Viewport = {
  width: "device-width", initialScale: 1, viewportFit: "cover", themeColor: "#18181b",
};

const NAV = [
  { href: "/", label: "Create" },
  { href: "/assets", label: "Asset Bible" },
  { href: "/chat", label: "Chat" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-zinc-100 text-zinc-900 antialiased">
        <header className="sticky top-0 z-30 border-b border-zinc-800 bg-zinc-900 text-zinc-100">
          <div className="mx-auto flex min-h-14 w-full max-w-6xl items-center gap-4 px-4">
            <Link href="/" className="text-[15px] font-bold tracking-tight">
              WREN <span className="text-amber-400">Story Engine</span>
            </Link>
            <nav className="flex items-center gap-1 text-sm">
              {NAV.map((n) => (
                <Link key={n.href} href={n.href}
                  className="flex min-h-11 items-center rounded-md px-3 text-zinc-300 hover:bg-zinc-800 hover:text-white">
                  {n.label}
                </Link>
              ))}
            </nav>
            <UserBadge />
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
