'use client';

import { useState } from 'react';
import Code from '@components/Code';
import CodeWrapper from '@components/CodeWrapper';
import { Link } from '@components/Link';
import ProjectButton from '@components/ProjectButton';
import ProjectMedia from '@components/ProjectMedia';
import ProjectTag from '@components/ProjectTag';
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

    return (
        <section className="max-w-3xl mx-auto flex flex-col p-8 gap-4 md:gap-8">
            {/* Header (Always visible) */}
            <div>
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
            <div className="flex border-b border-gray-light/30 mb-4">
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
                <div className="flex flex-col gap-8 animate-fadeIn duration-500">
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
                </div>
            ) : (
                <article className="prose sm:prose-base md:prose-lg lg:prose-xl xl:prose-2xl prose-p:text-gray-dark prose-headings:text-gray prose-a:text-blue marker:text-gray-dark animate-fadeIn duration-500">
                    {compiler(project.content, {
                        wrapper: null,
                        forceWrapper: true,
                        overrides: {
                            a: {
                                component: Link,
                            },
                            pre: {
                                component: CodeWrapper,
                            },
                            code: {
                                component: (props) => (
                                    <Code language="python" text={props.children} />
                                ),
                            },
                        },
                    })}
                </article>
            )}
        </section>
    );
}
