'use client';

import { useEffect, useState } from 'react';

interface TableOfContentsProps {
    content: string;
}

interface TocItem {
    id: string;
    text: string;
    level: number;
}

export default function TableOfContents({ content }: TableOfContentsProps) {
    const [headings, setHeadings] = useState<TocItem[]>([]);
    const [activeId, setActiveId] = useState<string>('');

    useEffect(() => {
        // Parse headings from markdown content
        const lines = content.split('\n');
        const extractedHeadings: TocItem[] = [];

        lines.forEach((line) => {
            // Match h2
            const match = line.match(/^(#{2})\s+(.+)$/);
            if (match) {
                const level = match[1].length;
                const text = match[2].trim();
                // Generate ID consistent with ProjectView
                const id = text.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
                extractedHeadings.push({ id, text, level });
            }
        });

        setHeadings(extractedHeadings);
    }, [content]);

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setActiveId(entry.target.id);
                    }
                });
            },
            { rootMargin: '-100px 0px -66% 0px' }
        );

        headings.forEach((heading) => {
            const element = document.getElementById(heading.id);
            if (element) {
                observer.observe(element);
            }
        });

        return () => observer.disconnect();
    }, [headings]);

    const handleClick = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
        e.preventDefault();
        const element = document.getElementById(id);
        if (element) {
            // Offset for fixed header
            const yOffset = -100;
            const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
            window.scrollTo({ top: y, behavior: 'smooth' });
            setActiveId(id);
        }
    };

    if (headings.length === 0) return null;

    return (
        <nav className="hidden lg:block w-64 shrink-0 order-first">
            <div className="sticky top-32 max-h-[calc(100vh-8rem)] overflow-y-auto pr-4">
                <h4 className="font-bold text-gray-dark mb-4 text-lg">Contents</h4>
                <ul className="space-y-2 border-l-2 border-gray-light/30">
                    {headings.map((heading) => (
                        <li key={heading.id} className={`pl-4 ${heading.level === 3 ? 'ml-4' : ''}`}>
                            <a
                                href={`#${heading.id}`}
                                onClick={(e) => handleClick(e, heading.id)}
                                className={`block text-sm transition-colors duration-200 ${activeId === heading.id
                                    ? 'text-blue font-bold border-l-2 border-blue -ml-[18px] pl-[14px]'
                                    : 'text-gray hover:text-blue/70'
                                    }`}
                            >
                                {heading.text}
                            </a>
                        </li>
                    ))}
                </ul>
            </div>
        </nav>
    );
}
