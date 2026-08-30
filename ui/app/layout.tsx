import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_ORIGIN ?? 'http://localhost:3000',
  ),
  title: 'TacoRank Run Monitor',
  description: 'Start and inspect autonomous TacoRank research runs.',
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
      <body>{children}</body>
    </html>
  );
}
