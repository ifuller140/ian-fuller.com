'use client';

import { useState } from 'react';
import { HiMenu, HiX } from 'react-icons/hi';
import { Link } from '@components/Link';
import { usePathname } from 'next/navigation';

export default function HamburgerMenu() {
    const [isOpen, setIsOpen] = useState(false);
    const pathname = usePathname();

    // Only show on home page
    if (pathname !== '/') {
        return null;
    }

    const toggleMenu = () => {
        setIsOpen(!isOpen);
    };

    return (
        <>
            {/* Hamburger Button */}
            <button
                onClick={toggleMenu}
                className="fixed top-24 right-8 z-50 p-3 bg-gray-dark text-white rounded-full shadow-lg hover:bg-blue transition-colors duration-300 md:hidden"
                aria-label="Open Menu"
            >
                {isOpen ? <HiX size={24} /> : <HiMenu size={24} />}
            </button>

            <button
                onClick={toggleMenu}
                className="fixed top-24 right-8 z-50 p-3 bg-gray-dark text-white rounded-full shadow-lg hover:bg-blue transition-colors duration-300 hidden md:block"
                aria-label="Open Menu"
            >
                {isOpen ? <HiX size={24} /> : <HiMenu size={24} />}
            </button>

            {/* Sidebar Overlay */}
            <div
                className={`fixed inset-0 bg-black/50 z-40 transition-opacity duration-300 ${isOpen ? 'opacity-100 visible' : 'opacity-0 invisible'
                    }`}
                onClick={toggleMenu}
            />

            {/* Sidebar */}
            <div
                className={`fixed top-0 right-0 h-full w-64 bg-white shadow-2xl z-50 transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : 'translate-x-full'
                    }`}
            >
                <div className="flex flex-col p-8 gap-8 h-full">
                    <div className="flex justify-end">
                        <button onClick={toggleMenu} className="text-gray-dark hover:text-blue">
                            <HiX size={32} />
                        </button>
                    </div>

                    <nav className="flex flex-col gap-6 text-xl font-bold text-gray-dark">
                        <Link href="/" onClick={toggleMenu} className="hover:text-blue">
                            Home
                        </Link>
                        <Link href="/#about" onClick={toggleMenu} className="hover:text-blue">
                            About
                        </Link>
                        <Link href="/#projects" onClick={toggleMenu} className="hover:text-blue">
                            Projects
                        </Link>
                        <Link href="/#contact" onClick={toggleMenu} className="hover:text-blue">
                            Contact
                        </Link>
                    </nav>

                    <div className="mt-auto text-sm text-gray-light">
                        <p>&copy; {new Date().getFullYear()} Ian Fuller</p>
                    </div>
                </div>
            </div>
        </>
    );
}
