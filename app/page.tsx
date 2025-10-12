import Hero from '@components/Hero';
import { Link } from '@components/Link';
import ProjectsSection from '@components/ProjectsSection';
import SkillsSection from '@components/SkillsSection';
import FeaturedProject from '@components/FeaturedProject';
import { Section } from '@components/Section';
import { Text } from '@components/Text';
import { getProjectMetadata } from '@util/ProjectMetadata';
import { GrDocumentUser } from 'react-icons/gr';
import { HiOutlineMail, HiOutlineUserAdd } from 'react-icons/hi';

export default function Home() {
  const projects = getProjectMetadata();
  const featuredProject = projects[0];

  return (
    <>
      <Hero />
      <div className="flex flex-col">
        <Section
          id="about"
          className="max-w-3xl m-auto flex flex-col gap-8 px-8 pt-14 pb-20"
        >
          <Text>
            I&apos;m a senior at the <strong>University of Maryland</strong>{' '}
            studying <strong>Computer Science</strong> with minors in{' '}
            <strong>Robotics</strong> and <strong>Entrepreneurship</strong>. I
            specialize in building systems that blend intelligent software with
            physical hardware—whether that&apos;s autonomous robots, computer
            vision pipelines, or full-stack applications.
          </Text>
          <Text>
            My work spans embedded systems, mobile development, simulation, and
            reinforcement learning. I&apos;ve built everything from{' '}
            <Link href="/project/smart-chess-board">chess-playing robots</Link>{' '}
            to document management systems used by real businesses. What excites
            me most is taking complex technical challenges and turning them into
            working, tangible solutions.
          </Text>
          <Text>
            I&apos;m graduating in <strong>May 2026</strong> and actively
            seeking full-time opportunities where I can apply my
            cross-disciplinary skill set. Whether it&apos;s robotics software
            engineering, full-stack development, or something in between— if
            you&apos;re building something ambitious, I&apos;d love to{' '}
            <Link href="/#contact">connect</Link>.
          </Text>
        </Section>

        <SkillsSection />

        <Section className="bg-white px-8 py-16 md:py-24">
          <div className="max-w-5xl mx-auto">
            <h2 className="font-extrabold text-2xl sm:text-3xl md:text-4xl text-center mb-12">
              Featured Project
            </h2>
            <FeaturedProject project={featuredProject} />
          </div>
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
            I&apos;m actively seeking{' '}
            <strong>full-time software engineering and robotics roles</strong>{' '}
            starting May 2026. Whether you&apos;re hiring, have an interesting
            project collaboration in mind, or just want to discuss autonomous
            systems, computer vision, or software development, I&apos;d love to
            hear from you!
          </Text>
          <div className="max-w-fit m-auto">
            <div className="grid sm:grid-cols-2 gap-8">
              <Link
                href="https://www.linkedin.com/in/ian-fuller-9a3932111/"
                target="_blank"
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
