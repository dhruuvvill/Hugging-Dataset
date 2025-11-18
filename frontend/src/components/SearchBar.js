'use client';

import { useState } from 'react';

export default function SearchBar({ onSubmit, loading }) {
  const [url, setUrl] = useState('');
  const [examples] = useState([
    'google/gemma-2-2b',
    'meta-llama/Llama-3.1-8B',
    'datasets/EleutherAI/hendrycks_math',
    'datasets/sentence-transformers/all-MiniLM-L6-v2',
    'openai/whisper-large-v3',
  ]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const fullUrl = url.startsWith('http') ? url : `https://huggingface.co/${url}`;
    onSubmit(fullUrl);
  };

  const handleExample = (example) => {
    setUrl(example);
    onSubmit(`https://huggingface.co/${example}`);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      {/* Google-style search box */}
      <div className="relative">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter HuggingFace model or dataset URL..."
          className="w-full px-5 py-4 text-lg rounded-full border border-gray-300 shadow-sm hover:shadow-md focus:outline-none focus:shadow-lg transition-shadow text-black"
          disabled={loading}
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex gap-2">
          <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* Google-style buttons */}
      <div className="flex justify-center gap-4 mt-6">
        <button
          type="submit"
          disabled={loading || !url}
          className="px-6 py-2.5 bg-gray-100 hover:bg-gray-200 border border-transparent rounded text-sm text-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Model Search
        </button>
        <button
          type="button"
          disabled={loading}
          className="px-6 py-2.5 bg-gray-100 hover:bg-gray-200 border border-transparent rounded text-sm text-gray-700 disabled:opacity-50"
          onClick={() => setUrl('')}
        >
          Clear
        </button>
      </div>

      {/* Examples */}
      <div className="mt-8 text-center">
        <p className="text-sm text-gray-600 mb-3">Suggested:</p>
        <div className="flex justify-center flex-wrap gap-2">
          {examples.slice(0, 3).map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => handleExample(example)}
              disabled={loading}
              className="px-4 py-2 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded text-sm text-blue-600 disabled:opacity-50 transition-colors"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </form>
  );
}

