'use client';

export default function FeatureCard({ title, value, description }) {
  const isBoolean = typeof value === 'boolean';
  const isNumber = typeof value === 'number';
  
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="text-sm font-medium text-gray-900 mb-1">{title}</h3>
          {description && (
            <p className="text-xs text-gray-500 mb-2">{description}</p>
          )}
          <div className="mt-2">
            {isBoolean ? (
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                value ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {value ? 'Yes' : 'No'}
              </span>
            ) : isNumber ? (
              <span className="text-lg font-semibold text-gray-900">{value.toLocaleString()}</span>
            ) : (
              <span className="text-sm text-gray-700">{String(value)}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

