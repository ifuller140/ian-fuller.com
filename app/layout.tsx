import Footer from '@components/Footer';
import NavBar from '@components/Navbar';
import '@styles/globals.css';

export const metadata = {
  title: 'Ian Fuller',
  description: "Ian Fuller's Portfolio Website",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white text-gray-dark flex flex-col">
        <NavBar />
        {children}
        <Footer />
      </body>
    </html>
  );
}
