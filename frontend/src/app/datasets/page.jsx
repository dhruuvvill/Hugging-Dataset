'use client';

import { useState, useEffect } from 'react';
import LoadingSpinner from '@/components/LoadingSpinner';
import DatasetCard from '@/components/DatasetCard';
import { getTopDatasets, getDatasetStats } from '@/lib/api';

export default function DatasetsPage() {
  const [topDatasets, setTopDatasets] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState('downloads');

  useEffect(() => {
    loadData();
  }, [sortBy]);

  const loadData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [top, statsData] = await Promise.all([
        getTopDatasets(10, sortBy),
        getDatasetStats(),
      ]);

      setTopDatasets(top.datasets || []);
      setStats(statsData);
    } catch (err) {
      setError(err.message || 'Failed to load dataset statistics');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Dataset <span className="text-blue-600">Statistics</span>
          </h1>
          <p className="text-gray-600 mt-2">
            Discover top-performing datasets on HuggingFace
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="max-w-2xl mx-auto mb-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex justify-center items-center py-12">
            <LoadingSpinner message="Loading dataset statistics..." />
          </div>
        )}

        {/* Stats Overview */}
        {stats && !loading && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <p className="text-sm text-gray-500 mb-1">Datasets Sampled</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total_datasets_sampled?.toLocaleString()}</p>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <p className="text-sm text-gray-500 mb-1">Avg Downloads</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats.average_downloads ? Math.round(stats.average_downloads).toLocaleString() : 'N/A'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <p className="text-sm text-gray-500 mb-1">Avg Likes</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats.average_likes ? Math.round(stats.average_likes).toLocaleString() : 'N/A'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <p className="text-sm text-gray-500 mb-1">Total Downloads</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats.total_downloads ? stats.total_downloads.toLocaleString() : 'N/A'}
              </p>
            </div>
          </div>
        )}

        {/* Sort By */}
        {!loading && (
          <div className="mb-6 flex items-center gap-4">
            <label className="text-sm font-medium text-gray-700">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
            >
              <option value="downloads">Downloads</option>
              <option value="likes">Likes</option>
            </select>
          </div>
        )}

        {/* Top 10 Datasets */}
        {!loading && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-6">
              Top 10 Datasets by {sortBy === 'downloads' ? 'Downloads' : 'Likes'}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {topDatasets.map((dataset, idx) => (
                <DatasetCard key={dataset.id} dataset={dataset} rank={idx + 1} />
              ))}
            </div>
            {topDatasets.length === 0 && (
              <div className="text-center py-12">
                <p className="text-gray-500">No top datasets found.</p>
              </div>
            )}
          </div>
        )}

        {/* Top Tags */}
        {stats && stats.top_tags && stats.top_tags.length > 0 && !loading && (
          <div className="mt-12 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Popular Tags</h3>
            <div className="flex flex-wrap gap-2">
              {stats.top_tags.map((item, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium"
                >
                  {item.tag} ({item.count})
                </span>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

