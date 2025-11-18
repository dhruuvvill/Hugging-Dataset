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
    <div className="min-h-screen bg-gray-50">
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Feature <span className="text-blue-600">Extraction</span>
          </h1>
          <p className="text-gray-600 mt-2">
            Extract and analyze comprehensive features from HuggingFace models
          </p>
        </div>

        {/* Search Section */}
        <div className="max-w-2xl mx-auto mb-8">
          <SearchBar onSubmit={handleExtract} loading={loading} />
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
            <LoadingSpinner message="Extracting features..." />
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="space-y-6">
            {/* Model Info & Prediction */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-semibold text-gray-900">{result.model_id}</h2>
                  {result.message && (
                    <p className="text-sm text-gray-600 mt-1">{result.message}</p>
                  )}
                </div>
                {result.prediction && (
                  <div className="text-right">
                    <div className={`inline-flex items-center px-4 py-2 rounded-lg font-semibold ${
                      result.prediction.is_popular 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-orange-100 text-orange-800'
                    }`}>
                      {result.prediction.is_popular ? 'High Popularity' : 'Low Popularity'}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Confidence: {(() => {
                        // Ensure confidence is properly calculated: if predicting low, use 1 - probability
                        const prob = result.prediction.is_popular 
                          ? result.prediction.probability 
                          : (1.0 - result.prediction.probability);
                        return (Math.max(0, Math.min(100, prob * 100))).toFixed(1);
                      })()}%
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Feature Importance */}
            {result.feature_importance && result.feature_importance.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
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
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Extracted Features</h3>
              <FeatureGrid 
                features={result.features} 
                featureImportance={result.feature_importance}
              />
            </div>
          </div>
        )}

        {/* Info Section */}
        {!result && !loading && (
          <div className="max-w-3xl mx-auto mt-12">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">About Feature Extraction</h3>
              <ul className="space-y-2 text-sm text-blue-800">
                <li>• Extracts 50+ features from model metadata including tags, licenses, and technical capabilities</li>
                <li>• Analyzes library usage, task types, research references, and language support</li>
                <li>• Provides popularity prediction based on trained machine learning model</li>
                <li>• Shows feature importance scores to understand what makes models popular</li>
              </ul>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
