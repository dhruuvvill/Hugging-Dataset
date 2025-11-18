'use client';

import Link from 'next/link';

export default function DatasetCard({ dataset, rank }) {
  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
  };

  const getTagColor = (tag) => {
    const colors = [
      'bg-blue-100 text-blue-800',
      'bg-green-100 text-green-800',
      'bg-purple-100 text-purple-800',
      'bg-yellow-100 text-yellow-800',
      'bg-pink-100 text-pink-800',
    ];
    const index = tag.length % colors.length;
    return colors[index];
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <span className="flex items-center justify-center w-8 h-8 bg-blue-100 text-blue-600 rounded-full font-semibold text-sm">
              {rank}
            </span>
            <Link
              href={`https://huggingface.co/datasets/${dataset.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-lg font-semibold text-gray-900 hover:text-blue-600 transition-colors"
            >
              {dataset.id}
            </Link>
          </div>
          <p className="text-sm text-gray-500 mb-3">by {dataset.author}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-xs text-gray-500 mb-1">Downloads</p>
          <p className="text-lg font-semibold text-gray-900">{formatNumber(dataset.downloads)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Likes</p>
          <p className="text-lg font-semibold text-gray-900">{formatNumber(dataset.likes)}</p>
        </div>
      </div>

      {dataset.tags && dataset.tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {dataset.tags.slice(0, 5).map((tag, idx) => (
            <span
              key={idx}
              className={`px-2 py-1 rounded-full text-xs font-medium ${getTagColor(tag)}`}
            >
              {tag}
            </span>
          ))}
          {dataset.tags.length > 5 && (
            <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
              +{dataset.tags.length - 5}
            </span>
          )}
        </div>
      )}

      {dataset.lastModified && (
        <p className="text-xs text-gray-400 mt-3">
          Updated: {new Date(dataset.lastModified).toLocaleDateString()}
        </p>
      )}
    </div>
  );
}

