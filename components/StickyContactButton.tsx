'use client';

import { Link } from '@components/Link';
import { useEffect, useState } from 'react';

export default function StickyContactButton() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const toggleVisibility = () => {
      // Show button after scrolling down 300px
      if (
        window.pageYOffset > 300 &&
        window.pageYOffset < (window.innerWidth < 600 ? 7000 : 5000)
      ) {
        setIsVisible(true);
      } else {
        setIsVisible(false);
      }
    };

    window.addEventListener('scroll', toggleVisibility);

    return () => window.removeEventListener('scroll', toggleVisibility);
  }, []);

  return (
    <>
      {isVisible && (
        <div className="fixed bottom-8 right-8 z-50 animate-fade-in">
          <Link href="/#contact" className="no-underline">
            <button className="bg-blue hover:bg-blue-dark text-white font-bold py-3 px-6 rounded-full shadow-2xl transition duration-300 ease-in-out hover:scale-110 flex items-center gap-2">
              <span className="text-sm md:text-base">Get in Touch</span>
              <span className="text-lg">→</span>
            </button>
          </Link>
        </div>
      )}
    </>
  );
}
