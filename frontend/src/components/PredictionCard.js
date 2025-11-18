'use client';

export default function PredictionCard({ data }) {
  const isHigh = data.predicted_popularity === 'high';
  // Ensure confidence is always between 0 and 100, and properly formatted
  const confidence = Math.max(0, Math.min(100, data.probability * 100));
  const confidencePercent = confidence.toFixed(1);
  
  // Detect if it's a dataset or model
  const isDataset = data.model_id.includes('/datasets/') || data.model_id.startsWith('datasets/');
  const entityType = isDataset ? 'dataset' : 'model';

  return (
    <div className="w-full bg-white">
      {/* Entity Info */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            {isDataset ? 'Dataset' : 'Model'}
          </span>
        </div>
        <h2 className="text-2xl text-gray-900 font-normal">{data.model_id}</h2>
        <a
          href={`https://huggingface.co/${data.model_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline text-sm"
        >
          View on HuggingFace →
        </a>
      </div>

      {/* Prediction Result - Google-style box */}
      <div className="border border-gray-200 rounded-lg p-6 mb-6">
        <div className="flex items-center gap-4 mb-4">
          <div className={`flex items-center justify-center w-12 h-12 rounded-full ${
            isHigh ? 'bg-green-100' : 'bg-red-100'
          }`}>
            {isHigh ? (
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
              {isHigh ? 'High' : 'Low'} Popularity
            </div>
            <div className="text-sm text-gray-600">
              {confidencePercent}% confidence
            </div>
          </div>
        </div>
        
        {/* Confidence bar - Google style */}
        <div className="w-full bg-gray-100 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full transition-all duration-500 ${
              isHigh ? 'bg-green-500' : 'bg-red-500'
            }`}
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
      </div>

      {/* Message */}
      {data.message && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-gray-700 leading-relaxed">{data.message}</p>
          </div>
        </div>
      )}

      {/* Features - Google style */}
      {/* {data.features && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Key Factors</h3>
          <div className="grid grid-cols-1 gap-2">
            {Object.entries(data.features).slice(0, 6).map(([key, value]) => (
              <div key={key} className="text-sm text-gray-600">
                <span className="text-gray-500">{key.replace(/_/g, ' ')}:</span> {value}
              </div>
            ))}
          </div>
        </div>
      )} */}
    </div>
  );
}

