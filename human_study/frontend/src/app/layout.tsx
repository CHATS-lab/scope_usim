import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "USIM Human Study",
  description: "Participant interface for the USIM multi-turn dialogue study.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text font-sans">{children}</body>
    </html>
  );
}
