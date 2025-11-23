import Footer from '@components/Footer';
import NavBar from '@components/Navbar';
import StickyContactButton from '@components/StickyContactButton';
import GoogleAnalytics from '@components/GoogleAnalytics';
import HamburgerMenu from '@components/HamburgerMenu';
import '@styles/globals.css';

export const metadata = {
  title: 'Ian Fuller | Software Engineer & Robotics Developer',
  description:
    "Ian Fuller's portfolio showcasing robotics, computer vision, and software engineering projects. UMD CS student graduating May 2026.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const GA_MEASUREMENT_ID = process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID;

  return (
    <html lang="en">
      <body className="min-h-screen bg-white text-gray-dark flex flex-col">
        {GA_MEASUREMENT_ID && (
          <GoogleAnalytics GA_MEASUREMENT_ID={GA_MEASUREMENT_ID} />
        )}

        <NavBar />
        <HamburgerMenu />
        {children}
        <StickyContactButton />
        <Footer />
      </body>
    </html>
  );
}
