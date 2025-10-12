import Hero from '@components/Hero';
import { Link } from '@components/Link';
import ProjectsSection from '@components/ProjectsSection';
import { Section } from '@components/Section';
import { Text } from '@components/Text';
import { getProjectMetadata } from '@util/ProjectMetadata';
import { GrDocumentUser } from 'react-icons/gr';
import { HiOutlineMail, HiOutlineUserAdd } from 'react-icons/hi';

export default function Home() {
  const projects = getProjectMetadata();

  return (
    <>
      <Hero />
      <div className="flex flex-col">
        <Section
          id="about"
          className="max-w-3xl m-auto flex flex-col gap-8 px-8 pt-14 pb-20"
        >
          <Text>
            I&apos;m a Computer Science and Robotics student at the University
            of Maryland graduating in May 2026. I specialize in autonomous
            systems and computer vision, bringing software and hardware together
            to solve complex problems.
          </Text>
          <Text>
            My work spans robotic manipulation, real-time vision processing, and
            embedded systems. I'm currently seeking full-time robotics and
            software engineering roles where I can apply my skills to build
            intelligent systems.
          </Text>
          <Text>
            Below are some projects that showcase my approach to both software
            and hardware challenges. Let's <Link href="/#contact">connect</Link>{' '}
            if you're building something ambitious.
          </Text>
        </Section>
        <ProjectsSection projects={projects} />
        <Section
          id="contact"
          className="max-w-3xl m-auto flex flex-col gap-8 px-8 py-16 md:py-24 lg:py-32 xl:py-40"
        >
          <div className="flex justify-center">
            <Text className="font-extrabold text-xl sm:text-2xl md:text-3xl lg:text-4xl xl:text-5xl">
              Contact Me
            </Text>
          </div>
          <Text>
            I&apos;m always excited to connect with fellow students,
            professionals, and anyone passionate about technology. Whether you
            have a question about my projects, want to collaborate, or just want
            to chat about the latest in tech, feel free to reach out!
          </Text>
          <div className="max-w-fit m-auto">
            <div className="grid sm:grid-cols-2 gap-8">
              <Link
                href="https://www.linkedin.com/in/ian-fuller-9a3932111/"
                className="text-gray-dark"
              >
                <div className="bg-white-light border border-1 border-white-dark h-full max-w-xs md:max-w-sm rounded-xl overflow-hidden shadow-lg transition duration-300 ease-in-out hover:-translate-y-1 hover:shadow-xl">
                  <div className="flex flex-col p-8 items-center">
                    <HiOutlineUserAdd className="text-3xl md:text-4xl lg:text-5xl xl:text-6xl" />
                    <h1 className="font-bold text-lg md:text-xl lg:text-2xl xl:text-3xl">
                      LinkedIn
                    </h1>
                    <p className="text-blue font-bold text-sm md:text-base lg:text-lg xl:text-xl">
                      Let&apos;s Connect!
                    </p>
                  </div>
                </div>
              </Link>
              <Link
                href="mailto:ianfuller140@gmail.com"
                className="text-gray-dark"
              >
                <div className="bg-white-light border border-1 border-white-dark h-full max-w-xs md:max-w-sm rounded-xl overflow-hidden shadow-lg transition duration-300 ease-in-out hover:-translate-y-1 hover:shadow-xl">
                  <div className="flex flex-col p-8 items-center">
                    <HiOutlineMail className="text-3xl md:text-4xl lg:text-5xl xl:text-6xl" />
                    <h1 className="font-bold text-lg md:text-xl lg:text-2xl xl:text-3xl">
                      Email
                    </h1>
                    <p className="text-blue font-bold text-sm md:text-base lg:text-lg xl:text-xl">
                      ianfuller140@gmail.com
                    </p>
                  </div>
                </div>
              </Link>
            </div>
          </div>
        </Section>
      </div>
    </>
  );
}
