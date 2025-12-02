'use client';

import { useState } from 'react';
import Code from '@components/Code';
import CodeWrapper from '@components/CodeWrapper';
import { Link } from '@components/Link';
import ProjectButton from '@components/ProjectButton';
import ProjectMedia from '@components/ProjectMedia';
import ProjectTag from '@components/ProjectTag';
import TableOfContents from '@components/TableOfContents';
import { compiler } from 'markdown-to-jsx';

interface ProjectViewProps {
    project: {
        data: {
            title: string;
            tags?: string[];
            media?: string;
            image: string;
            links?: { text: string; href: string }[];
            description?: string;
        };
        content: string;
    };
}

export default function ProjectView({ project }: ProjectViewProps) {
    const [activeTab, setActiveTab] = useState<'overview' | 'technical'>('overview');

    // Split content based on "<!-- split -->" delimiter
    const splitDelimiter = '<!-- split -->';
    const splitIndex = project.content.indexOf(splitDelimiter);
    let overviewContent = '';
    let technicalContent = '';

    if (splitIndex !== -1) {
        overviewContent = project.content.substring(0, splitIndex);
        technicalContent = project.content.substring(splitIndex + splitDelimiter.length);
    } else {
        // Fallback: everything in overview if split point not found
        overviewContent = project.content;
    }

    // Custom ID generation for headers to support TOC
    const HeadingRenderer = ({ level, children, ...props }: any) => {
        const id = children[0]?.toString().toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
        const Tag = `h${level}` as keyof JSX.IntrinsicElements;
        return <Tag id={id} {...props}>{children}</Tag>;
    };

    return (
        <section className="max-w-7xl mx-auto flex flex-col p-8 gap-4 md:gap-8">
            {/* Header (Always visible) */}
            <div className="max-w-3xl mx-auto w-full">
                <h1 className="font-extrabold text-2xl sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl pb-2 md:pb-4">
                    {project.data.title}
                </h1>
                <div className="flex flex-wrap gap-2">
                    {project.data.tags &&
                        project.data.tags.map((tag: string) => (
                            <ProjectTag key={tag} tag={tag} />
                        ))}
                </div>
            </div>

            {/* Tabs */}
            <div className="max-w-3xl mx-auto w-full flex border-b border-gray-light/30 mb-4">
                <button
                    className={`py-2 px-4 font-bold text-lg transition-colors duration-300 ${activeTab === 'overview'
                        ? 'text-blue border-b-2 border-blue'
                        : 'text-gray hover:text-blue/70'
                        }`}
                    onClick={() => setActiveTab('overview')}
                >
                    Overview
                </button>
                <button
                    className={`py-2 px-4 font-bold text-lg transition-colors duration-300 ${activeTab === 'technical'
                        ? 'text-blue border-b-2 border-blue'
                        : 'text-gray hover:text-blue/70'
                        }`}
                    onClick={() => setActiveTab('technical')}
                >
                    Technical Details
                </button>
            </div>

            {/* Content */}
            {activeTab === 'overview' ? (
                <div className="max-w-3xl mx-auto w-full flex flex-col gap-8 animate-fadeIn duration-500">

                    <ProjectMedia
                        title={project.data.title}
                        media={project.data.media || ''}
                        image={project.data.image}
                    />

                    <div className="flex flex-wrap gap-4">
                        {project.data.links &&
                            project.data.links.map((link: { text: string; href: string }) => (
                                <ProjectButton key={link.text} href={link.href}>
                                    {link.text}
                                </ProjectButton>
                            ))}
                    </div>

                    {project.data.description && (
                        <div className="text-lg md:text-xl text-gray-dark">
                            <p>{project.data.description}</p>
                        </div>
                    )}

                    <article className="prose sm:prose-base md:prose-lg lg:prose-xl xl:prose-2xl prose-p:text-gray-dark prose-headings:text-gray prose-a:text-blue marker:text-gray-dark">
                        {compiler(overviewContent, {
                            wrapper: null,
                            forceWrapper: true,
                            overrides: {
                                a: { component: Link },
                                pre: { component: CodeWrapper },
                                video: { component: (props) => <video playsInline {...props} /> },
                            },
                        })}
                    </article>

                    {/* View Technical Details Button */}
                    <div className="flex justify-center mt-4">
                        <button
                            className="bg-red hover:bg-red/80 text-white font-bold rounded-full px-4 py-2 text-lg md:text-xl transition-all duration-300 flex items-center gap-2 hover:gap-3"
                            onClick={() => {
                                setActiveTab('technical');
                                window.scrollTo({ top: 0, behavior: 'smooth' });
                            }}
                        >
                            View Technical Details &rarr;
                        </button>
                    </div>
                </div>
            ) : (
                <div className="flex flex-col lg:flex-row gap-8 animate-fadeIn duration-500 relative">
                    {/* Table of Contents Sidebar */}
                    <TableOfContents content={technicalContent} />

                    {/* Main Content */}
                    <article className="flex-1 prose sm:prose-base md:prose-lg lg:prose-xl xl:prose-2xl prose-p:text-gray-dark prose-headings:text-gray prose-a:text-blue marker:text-gray-dark max-w-3xl mx-auto lg:mx-0">
                        {compiler(technicalContent, {
                            wrapper: null,
                            forceWrapper: true,
                            overrides: {
                                a: { component: Link },
                                pre: { component: CodeWrapper },
                                h1: { component: (props) => <HeadingRenderer level={1} {...props} /> },
                                h2: { component: (props) => <HeadingRenderer level={2} {...props} /> },
                                h3: { component: (props) => <HeadingRenderer level={3} {...props} /> },
                                video: { component: (props) => <video playsInline {...props} /> },
                            },
                        })}
                    </article>
                </div>
            )}
        </section>
    );
}
