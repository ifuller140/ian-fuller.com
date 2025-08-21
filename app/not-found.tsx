import Link from 'next/link';

export default function NotFound() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-white-light text-gray text-center px-6">
      <div className="text-7xl mb-3 animate-bounce">🐾</div>
      <h1 className="text-4xl font-semibold mb-2">404: Lost in Space!</h1>
      <p className="text-lg mb-6">
        Oops! The page you’re looking for wandered off.
        <br />
        Let’s get you back <strong>home</strong>!
      </p>
      <Link
        href="/"
        className="inline-block rounded-full px-6 py-3 bg-red text-[#ffffff] font-semibold transition-colors hover:bg-[#A53046]"
      >
        Take me home
      </Link>
    </main>
  );
}
