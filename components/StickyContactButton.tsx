'use client';

import { Link } from '@components/Link';
import { useEffect, useState } from 'react';

export default function StickyContactButton() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const toggleVisibility = () => {
      // Show button after scrolling down 300px
      if (window.scrollY > 300) {
        setIsVisible(true);
      } else {
        setIsVisible(false);
      }
    };

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(false);
          } else if (window.scrollY > 300) {
            setIsVisible(true);
          }
        });
      },
      { threshold: 0.1 }
    );

    const contactSection = document.getElementById('contact');
    if (contactSection) {
      observer.observe(contactSection);
    }

    window.addEventListener('scroll', toggleVisibility);

    return () => {
      window.removeEventListener('scroll', toggleVisibility);
      observer.disconnect();
    };
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
