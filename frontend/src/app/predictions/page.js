'use client';

import { useState } from 'react';
import SearchBar from '@/components/SearchBar';
import PredictionCard from '@/components/PredictionCard';
import LoadingSpinner from '@/components/LoadingSpinner';
import { predictModel } from '@/lib/api';

export default function PredictionsPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (url) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictModel(url);
      setResult(prediction);
    } catch (err) {
      setError(err.message || 'Failed to predict model popularity');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-transparent relative">
      <main className="flex flex-col items-center justify-center min-h-screen relative z-10">
        {/* Logo/Title */}
        <div className="text-center mb-8">
          <h1 className="text-6xl font-normal text-gray-900 mb-2 tracking-tight">
            Model & Dataset
            <span className="text-blue-600"> Popularity</span>
          </h1>
          <p className="text-sm text-gray-500 mt-2">Predict popularity of HuggingFace models and datasets</p>
        </div>

        {/* Search Section */}
        <div className="w-full max-w-2xl px-4">
          <SearchBar onSubmit={handlePredict} loading={loading} />
        </div>

        {/* Error Message */}
        {error && (
          <div className="max-w-2xl w-full mt-6 px-4">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="mt-12">
            <LoadingSpinner message="Analyzing model popularity..." />
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="w-full max-w-2xl mt-8 px-4">
            <PredictionCard data={result} />
          </div>
        )}

        {/* Footer - Google style */}
        <div className="absolute bottom-0 left-0 right-0">
          <div className="flex justify-center gap-6 py-4 text-gray-500 text-sm">
            <span>Powered by ML</span>
            <span>•</span>
            <span>604K+ models trained</span>
            <span>•</span>
            <span>70%+ accuracy</span>
          </div>
        </div>
      </main>
    </div>
  );
}

