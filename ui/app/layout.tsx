import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_ORIGIN ?? 'http://localhost:3000',
  ),
  title: 'TacoRank Run Monitor',
  description: 'Live, read-only monitoring for autonomous TacoRank research runs.',
  openGraph: {
    title: 'TacoRank Run Monitor',
    description: 'Autonomous research, observed live.',
    type: 'website',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'TacoRank Run Monitor' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'TacoRank Run Monitor',
    description: 'Autonomous research, observed live.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
