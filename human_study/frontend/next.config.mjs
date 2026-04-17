const apiBase = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${apiBase}/:path*` },
    ];
  },
};

export default nextConfig;
