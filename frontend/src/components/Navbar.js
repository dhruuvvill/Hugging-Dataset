'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();

  const isActive = (path) => {
    return pathname === path;
  };

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo/Brand */}
          <div className="flex-shrink-0">
            <Link href="/" className="flex items-center">
              <span className="text-xl font-semibold text-gray-900">
                HuggingFace
                <span className="text-blue-600"> Analyzer</span>
              </span>
            </Link>
          </div>

          {/* Navigation Links */}
          <div className="flex space-x-8">
            <Link
              href="/"
              className={`inline-flex items-center px-3 py-2 text-sm font-medium transition-colors ${
                isActive('/')
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:text-gray-900 hover:border-b-2 hover:border-gray-300'
              }`}
            >
              Feature Extraction
            </Link>
            <Link
              href="/predictions"
              className={`inline-flex items-center px-3 py-2 text-sm font-medium transition-colors ${
                isActive('/predictions')
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:text-gray-900 hover:border-b-2 hover:border-gray-300'
              }`}
            >
              Predictions
            </Link>
            <Link
              href="/datasets"
              className={`inline-flex items-center px-3 py-2 text-sm font-medium transition-colors ${
                isActive('/datasets')
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:text-gray-900 hover:border-b-2 hover:border-gray-300'
              }`}
            >
             Stats
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

