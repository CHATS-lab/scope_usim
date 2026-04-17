import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "USIM Human Study",
  description: "Participant interface for the USIM multi-turn dialogue study.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text font-sans antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-accent focus:px-3 focus:py-1 focus:text-bg"
        >
          Skip to main content
        </a>
        <div id="main">{children}</div>
      </body>
    </html>
  );
}
