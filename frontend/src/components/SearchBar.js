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
          className="w-full px-5 py-4 text-lg rounded-full border border-white/30 bg-white/80 backdrop-blur-md shadow-lg hover:shadow-xl focus:outline-none focus:shadow-2xl focus:bg-white/90 transition-all text-black"
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
          className="px-6 py-2.5 bg-white/70 backdrop-blur-md hover:bg-white/90 border border-white/30 rounded text-sm text-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg"
        >
          Model Search
        </button>
        <button
          type="button"
          disabled={loading}
          className="px-6 py-2.5 bg-white/70 backdrop-blur-md hover:bg-white/90 border border-white/30 rounded text-sm text-gray-700 disabled:opacity-50 transition-all shadow-md hover:shadow-lg"
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
              className="px-4 py-2 bg-white/60 backdrop-blur-md hover:bg-white/80 border border-white/30 rounded text-sm text-blue-600 disabled:opacity-50 transition-all shadow-sm hover:shadow-md"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </form>
  );
}

