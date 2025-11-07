export default function SkillsSection() {
  const skillCategories = [
    {
      title: 'Languages',
      skills: [
        'Python',
        'C++',
        'Java',
        'TypeScript',
        'JavaScript',
        'C',
        'MATLAB',
      ],
    },
    {
      title: 'Robotics & Hardware',
      skills: [
        'ROS',
        'Gazebo',
        'OpenCV',
        'Raspberry Pi',
        'Arduino',
        'Isaac Sim',
      ],
    },
    {
      title: 'Software & Frameworks',
      skills: ['React Native', 'NumPy', 'PyTorch', 'AWS', 'Docker', 'Git'],
    },
    {
      title: 'Specialties',
      skills: [
        'Computer Vision',
        'Embedded Systems',
        'Autonomous Systems',
        'Full-Stack Dev',
        'Simulation',
        'ML/RL',
      ],
    },
  ];

  return (
    <section className="bg-white-light px-8 py-16 md:py-20">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-extrabold text-2xl sm:text-3xl md:text-4xl text-center mb-12">
          Technical Skills
        </h2>
        <div className="grid md:grid-cols-2 gap-8">
          {skillCategories.map((category) => (
            <div
              key={category.title}
              className="bg-white rounded-xl p-6 shadow-md border border-white-dark"
            >
              <h3 className="font-bold text-lg md:text-xl lg:text-2xl mb-4 text-gray">
                {category.title}
              </h3>
              <div className="flex flex-wrap gap-2">
                {category.skills.map((skill) => (
                  <span
                    key={skill}
                    className="inline-block bg-white-light text-gray-dark border border-gray-light rounded-full px-3 py-1 text-sm md:text-base font-medium"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
