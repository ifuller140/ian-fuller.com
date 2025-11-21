import ProjectView from '@components/ProjectView';
import { getProjectMetadata } from '@util/ProjectMetadata';
import fs from 'fs';
import matter from 'gray-matter';

const getProjectContent = (id: string) => {
  const folder = 'content/';
  const file = `${folder}${id}.md`;
  const content = fs.readFileSync(file, 'utf8');
  return matter(content);
};

export const generateStaticParams = async () => {
  const projects = getProjectMetadata();
  return projects.map((project) => ({
    id: project.id,
  }));
};

export default function Project(props: any) {
  const id = props.params.id;
  const project = getProjectContent(id);

  // Serialize the project data to pass to Client Component
  const serializedProject = {
    data: project.data as any,
    content: project.content
  };

  return <ProjectView project={serializedProject} />;
}
