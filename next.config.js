/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: true,
  },
  images: {
    domains: ['picsum.photos', 'via.placeholder.com'],
  },
}

module.exports = nextConfig
