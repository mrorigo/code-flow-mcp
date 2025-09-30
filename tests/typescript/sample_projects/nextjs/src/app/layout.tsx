import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';
import { Navigation } from '@/components/Navigation';
import { Toaster } from '@/components/ui/Toaster';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: {
    default: 'Next.js TypeScript App',
    template: '%s | Next.js TypeScript App'
  },
  description: 'A comprehensive Next.js application with TypeScript, demonstrating modern React patterns and TypeScript features.',
  keywords: ['Next.js', 'TypeScript', 'React', 'Tailwind CSS'],
  authors: [{ name: 'Developer' }],
  creator: 'Developer',
  publisher: 'Developer',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('http://localhost:3000'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'http://localhost:3000',
    title: 'Next.js TypeScript App',
    description: 'A comprehensive Next.js application with TypeScript',
    siteName: 'Next.js TypeScript App',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Next.js TypeScript App',
    description: 'A comprehensive Next.js application with TypeScript',
    creator: '@developer',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} antialiased`}>
        <Providers>
          <div className="min-h-screen bg-background font-sans antialiased">
            <Navigation />
            <main className="flex-1">
              {children}
            </main>
            <footer className="border-t py-6 md:px-8 md:py-0">
              <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
                <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
                  <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
                    Built with{' '}
                    <a
                      href="https://nextjs.org"
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium underline underline-offset-4"
                    >
                      Next.js
                    </a>
                    ,{' '}
                    <a
                      href="https://www.typescriptlang.org"
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium underline underline-offset-4"
                    >
                      TypeScript
                    </a>
                    , and{' '}
                    <a
                      href="https://tailwindcss.com"
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium underline underline-offset-4"
                    >
                      Tailwind CSS
                    </a>
                    .
                  </p>
                </div>
              </div>
            </footer>
          </div>
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}