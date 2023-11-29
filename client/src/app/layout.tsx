import type { Metadata, Viewport } from 'next';

import Providers from '~/app/providers';
import Layout from '~/lib/layout';

type RootLayoutProps = {
  children: React.ReactNode;
};

const APP_NAME = 'RedCam - VideoSD - redcam.redmond.ai';

export const metadata: Metadata = {
  title: { default: APP_NAME, template: '%s' },
  description: 'StableDiffusion over webRTC',
  applicationName: APP_NAME,
  appleWebApp: {
    capable: true,
    title: APP_NAME,
    statusBarStyle: 'default',
  },
  formatDetection: {
    telephone: false,
  },
  openGraph: {
    url: 'https://redcam.redmond.ai/og.png',
    title: 'RedCam',
    description: 'StableDiffusion over webRTC',
    images: {
      url: 'https://redcam.redmond.ai/og.png',
      alt: 'redcam.redmond.ai',
    },
  },
  twitter: {
    creator: '@venetanji',
    card: 'summary_large_image',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#FFFFFF',
};

const RootLayout = ({ children }: RootLayoutProps) => {
  return (
    <html lang="en">
      <body>
        <Providers>
          <Layout>{children}</Layout>
        </Providers>
      </body>
    </html>
  );
};

export default RootLayout;
