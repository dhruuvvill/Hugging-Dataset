'use client';

import { useState } from 'react';
import SearchBar from '@/components/SearchBar';
import LoadingSpinner from '@/components/LoadingSpinner';
import FeatureGrid from '@/components/FeatureGrid';
import { extractFeatures } from '@/lib/api';

export default function Home() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleExtract = async (url) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await extractFeatures(url);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to extract features');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <main className="flex flex-col items-center justify-center min-h-screen">
        {/* Logo/Title */}
        <div className="text-center mb-8">
          <h1 className="text-6xl font-normal text-gray-900 mb-2 tracking-tight">
            Feature <span className="text-blue-600">Extraction</span>
          </h1>
          <p className="text-sm text-gray-500 mt-2">Extract and analyze comprehensive features from HuggingFace models and datasets</p>
        </div>

        {/* Search Section */}
        <div className="w-full max-w-2xl px-4">
          <SearchBar onSubmit={handleExtract} loading={loading} />
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
            <LoadingSpinner message="Extracting features..." />
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="w-full max-w-7xl mt-8 px-4 pb-20">
            <div className="space-y-6">
              {/* Model Info & Prediction */}
              <div className="bg-white border border-gray-200 rounded-lg p-6">
                <div className="mb-6">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                      {result.model_id.includes('/datasets/') || result.model_id.startsWith('datasets/') ? 'Dataset' : 'Model'}
                    </span>
                  </div>
                  <h2 className="text-2xl text-gray-900 font-normal">{result.model_id}</h2>
                  <a
                    href={`https://huggingface.co/${result.model_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline text-sm"
                  >
                    View on HuggingFace →
                  </a>
                </div>
                {result.message && (
                  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-start gap-3">
                      <svg className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <p className="text-sm text-gray-700 leading-relaxed">{result.message}</p>
                    </div>
                  </div>
                )}
                {result.prediction && (
                  <div className="border border-gray-200 rounded-lg p-6">
                    <div className="flex items-center gap-4 mb-4">
                      <div className={`flex items-center justify-center w-12 h-12 rounded-full ${
                        result.prediction.is_popular ? 'bg-green-100' : 'bg-red-100'
                      }`}>
                        {result.prediction.is_popular ? (
                          <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        ) : (
                          <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        )}
                      </div>
                      <div>
                        <div className="text-xl font-medium text-gray-900">
                          {result.prediction.is_popular ? 'High' : 'Low'} Popularity
                        </div>
                        <div className="text-sm text-gray-600">
                          {(() => {
                            const prob = result.prediction.is_popular 
                              ? result.prediction.probability 
                              : (1.0 - result.prediction.probability);
                            return (Math.max(0, Math.min(100, prob * 100))).toFixed(1);
                          })()}% confidence
                        </div>
                      </div>
                    </div>
                    <div className="w-full bg-gray-100 rounded-full h-2.5">
                      <div
                        className={`h-2.5 rounded-full transition-all duration-500 ${
                          result.prediction.is_popular ? 'bg-green-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${(() => {
                          const prob = result.prediction.is_popular 
                            ? result.prediction.probability 
                            : (1.0 - result.prediction.probability);
                          return Math.max(0, Math.min(100, prob * 100));
                        })()}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Feature Importance */}
              {result.feature_importance && result.feature_importance.length > 0 && (
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Feature Importance</h3>
                  <div className="space-y-2">
                    {result.feature_importance.slice(0, 10).map((item, idx) => (
                      <div key={idx} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
                        <span className="text-sm text-gray-700">
                          {item.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                        <div className="flex items-center gap-3">
                          <div className="w-32 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ width: `${(item.importance * 100)}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium text-gray-900 w-16 text-right">
                            {(item.importance * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Features Grid */}
              <div className="bg-white border border-gray-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Extracted Features</h3>
                <FeatureGrid 
                  features={result.features} 
                  featureImportance={result.feature_importance}
                />
              </div>
            </div>
          </div>
        )}

        {/* Footer - Google style */}
        <div className="absolute bottom-0 left-0 right-0">
          <div className="flex justify-center gap-6 py-4 text-gray-500 text-sm">
            <span>Powered by ML</span>
            <span>•</span>
            <span>50+ features extracted</span>
            <span>•</span>
            <span>Real-time analysis</span>
          </div>
        </div>
      </main>
    </div>
  );
}
