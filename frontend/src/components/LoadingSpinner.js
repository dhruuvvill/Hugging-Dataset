'use client';

export default function LoadingSpinner({ message = "Analyzing model..." }) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-8 h-8">
        <div className="absolute inset-0 border-2 border-gray-300 rounded-full"></div>
        <div className="absolute inset-0 border-2 border-blue-500 rounded-full border-t-transparent animate-spin"></div>
      </div>
      <p className="text-sm text-gray-500">{message}</p>
    </div>
  );
}

