import type { NextConfig } from "next";

const internalApiUrl = process.env.COGNEE_INTERNAL_API_URL || "http://api:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/cognee-api/:path*",
        destination: `${internalApiUrl}/:path*`,
      },
    ];
  },
  images: {
    remotePatterns: [{
      protocol: "https",
      hostname: "lh3.googleusercontent.com",
    }],
  },
};

export default nextConfig;
