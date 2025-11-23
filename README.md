[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/ifuller140/ian-fuller.com/blob/master/LICENSE)
[![Next.js CI/CD](https://github.com/ifuller140/ian-fuller.com/actions/workflows/nextjs.yml/badge.svg)](https://github.com/ifuller140/ian-fuller.com/actions/workflows/nextjs.yml)

# [ian-fuller.com](https://ian-fuller.com)

My personal portfolio website showcasing projects in robotics, computer vision, and software engineering. Built with Next.js 14+, TypeScript, and Tailwind CSS.

## About

This portfolio serves as a comprehensive showcase of my work spanning:

- **Robotics & Autonomous Systems**: ROS-based projects, embedded systems, and hardware integration
- **Computer Vision**: Image processing, object detection, and real-time perception pipelines
- **Full-Stack Development**: Web and mobile applications with modern frameworks

The site features an interactive boids simulation on the hero section, dynamic project cards with hover previews, and a responsive design optimized for all devices.

## Table of Contents

- [Installation](#installation)
- [Deployment](#deployment)
- [Development](#development)
- [Issues](#issues)
- [License](#license)

## Installation

This project requires [Node.js](https://nodejs.org) (v20+) and npm.

```bash
# Clone the repository
git clone https://github.com/ifuller140/ian-fuller.com.git
cd ian-fuller.com

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the site.

## Deployment

### GitHub Pages

This site is configured for static export and GitHub Pages deployment.

```bash
# Build static site
npm run build

# Files are exported to /out directory
```

The included GitHub Actions workflow (`.github/workflows/nextjs.yml`) automatically deploys to GitHub Pages on push to `main`.

## Development

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

### Code Style

This project uses:

- **ESLint**: For code linting
- **Prettier**: For code formatting (see `.prettierrc.json`)
- **TypeScript**: For type safety

## Issues

To report a bug or request a feature, go to the [Issues page](https://github.com/ifuller140/ian-fuller.com/issues).

## License

Distributed under the [MIT License](https://github.com/ifuller140/ian-fuller.com/blob/main/LICENSE).

## Acknowledgments

- Next.js for the framework
- Tailwind CSS for styling
- Boids algorithm inspired by Craig Reynolds
- All the open-source libraries that made this possible

---

### Feedback

I'd love to hear any feedback you have! Feel free to reach out via [email](mailto:ianfuller140@gmail.com).

---

**Built by Ian Fuller** • [Website](https://ian-fuller.com) • [GitHub](https://github.com/ifuller140) • [LinkedIn](https://www.linkedin.com/in/ian-fuller-9a3932111/)
