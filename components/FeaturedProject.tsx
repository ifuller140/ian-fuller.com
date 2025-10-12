import { ProjectMetadata } from '@util/ProjectMetadata';
import Image from 'next/image';
import Link from 'next/link';
import ProjectTag from '@components/ProjectTag';

export interface FeaturedProjectProps {
  project: ProjectMetadata;
}

export default function FeaturedProject({ project }: FeaturedProjectProps) {
  return (
    <Link href={'/project/' + project.id}>
      <div className="bg-white rounded-3xl overflow-hidden shadow-xl transition duration-300 ease-in-out hover:-translate-y-2 hover:shadow-2xl border border-white-dark">
        <div className="grid md:grid-cols-2 gap-0">
          {/* Image Section */}
          <div
            className="relative aspect-video md:aspect-auto md:min-h-[400px]"
            style={{ backgroundColor: project.bgColor || '#F5F5F5' }}
          >
            {
              <Image
                src={'/' + project.image}
                alt={project.title}
                fill
                className="object-cover"
              />
            }
          </div>

          {/* Content Section */}
          <div className="flex flex-col justify-center p-8 md:p-10 lg:p-12">
            <h3 className="font-bold text-2xl md:text-3xl lg:text-4xl mb-4">
              {project.title}
            </h3>
            <p className="text-base md:text-lg lg:text-xl text-gray-dark mb-6">
              {project.description}
            </p>
            <div className="flex flex-wrap gap-2 mb-4 md:mb-14">
              {project.tags &&
                project.tags.map((tag) => <ProjectTag key={tag} tag={tag} />)}
            </div>
            <div className="inline-block">
              <span className="text-blue font-semibold text-base md:text-lg">
                View Project Details →
              </span>
            </div>
          </div>
        </div>
      </div>
    </Link>
  );
}
