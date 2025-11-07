'use client';

import ProjectTag from '@components/ProjectTag';
import { ProjectMetadata } from '@util/ProjectMetadata';
import isMobile from '@util/isMobile';
import Image from 'next/image';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { useIntersectionObserver } from 'usehooks-ts';

export interface CardProps {
  project: ProjectMetadata;
  previewed: boolean;
  onFocusChange: (focused: boolean, id: string) => void;
}

export default function Card({ project, previewed, onFocusChange }: CardProps) {
  const [focused, setFocused] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [previewLoaded, setPreviewLoaded] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const ref = useRef<HTMLDivElement | null>(null);
  const entry = useIntersectionObserver(ref, { threshold: 1 });
  const isVisible = !!entry?.isIntersecting;
  const [debouncedIsVisible, setDebouncedIsVisible] = useState(isVisible);

  // Check if preview is a video
  const isVideoPreview =
    project.preview?.endsWith('.mp4') || project.preview?.endsWith('.webm');

  // Preload the preview GIF when card becomes visible
  useEffect(() => {
    if (project.preview && isVisible && !previewLoaded && !isVideoPreview) {
      const img = new window.Image();
      img.src = '/' + project.preview;
      img.onload = () => setPreviewLoaded(true);
    }
  }, [project.preview, isVisible, previewLoaded, isVideoPreview]);

  // Handle video playback
  useEffect(() => {
    if (videoRef.current && isVideoPreview) {
      if (previewed) {
        videoRef.current.play().catch(() => {
          // Autoplay failed (browser policy), this is okay
        });
      } else {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
      }
    }
  }, [previewed, isVideoPreview]);

  // Debounce is visible if changing from not visible to visible
  useEffect(() => {
    const timer = setTimeout(
      () => setDebouncedIsVisible(isVisible),
      isVisible ? 1500 : 0
    );

    return () => {
      clearTimeout(timer);
    };
  }, [isVisible]);

  useEffect(() => {
    // If no preview, don't focus
    if (!project.preview) {
      setFocused(false);
    }
    // If on mobile, focus if fully visible
    else if (isMobile()) {
      setFocused(debouncedIsVisible);
    }
    // Otherwise, focus if hovered
    else {
      setFocused(hovered);
    }
  }, [project, hovered, debouncedIsVisible, isMobile]);

  useEffect(() => {
    onFocusChange(focused, project.id);
  }, [focused]);

  return (
    <Link href={'/project/' + project.id}>
      <div
        className="bg-white h-full max-w-xs md:max-w-sm rounded-3xl overflow-hidden shadow-xl transition duration-300 ease-in-out hover:-translate-y-4 hover:-translate-x-0.5 hover:shadow-2xl"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        ref={ref}
      >
        <div
          className="relative aspect-5/4"
          style={{ backgroundColor: project.bgColor }}
        >
          {project.preview && previewed ? (
            isVideoPreview ? (
              <video
                ref={videoRef}
                src={'/' + project.preview}
                className="w-full h-full object-contain"
                loop
                muted
                playsInline
                preload="auto"
              />
            ) : previewLoaded ? (
              <Image
                src={'/' + project.preview}
                alt={project.title}
                fill
                className="object-contain"
              />
            ) : (
              <>
                <Image
                  src={'/' + project.image}
                  alt={project.title}
                  fill
                  className="object-cover"
                />
                {/* Loading indicator */}
                <div className="absolute inset-5 flex items-center justify-center bg-black bg-opacity-20">
                  <div className="animate-spin rounded-full h-20 w-20 border-b-2 border-gray"></div>
                </div>
              </>
            )
          ) : (
            <Image
              src={'/' + project.image}
              alt={project.title}
              fill
              className="object-cover"
            />
          )}

          {/* Subtle hover hint that only shows on desktop when preview exists */}
          {project.preview && !isMobile() && !previewed && (
            <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-5 transition-all duration-300 flex items-center justify-center">
              <div
                className={`text-white text-sm bg-gray bg-opacity-70 px-3 py-1 rounded-full opacity-0 hover:opacity-100 transition-opacity duration-300`}
              >
                Hover to preview
              </div>
            </div>
          )}
        </div>
        <div className="flex flex-col gap-4 p-6">
          <div>
            <h1 className="font-bold text-base md:text-lg lg:text-xl xl:text-2xl">
              {project.title}
            </h1>
            <p className="text-sm md:text-base lg:text-lg xl:text-xl">
              {project.description}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {project.tags &&
              project.tags.map((tag) => <ProjectTag key={tag} tag={tag} />)}
          </div>
        </div>
      </div>
    </Link>
  );
}
