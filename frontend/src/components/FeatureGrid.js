'use client';

import FeatureCard from './FeatureCard';

export default function FeatureGrid({ features, featureImportance }) {
  // Group features by category
  const categories = {
    'Basic': ['num_tags', 'days_since_modification', 'is_recent', 'is_very_recent'],
    'Libraries': ['has_transformers', 'has_safetensors', 'has_diffusers', 'has_sentence_transformers', 'has_gguf', 'has_lora'],
    'Tasks': ['has_text_generation', 'has_image_to_text', 'has_text_to_image', 'has_image_to_image', 'has_text_to_video', 'has_ocr', 'has_conversational', 'has_feature_extraction'],
    'Technical': ['has_custom_code', 'has_autotrain_compatible', 'has_endpoints_compatible', 'has_text_generation_inference'],
    'Base Models': ['num_base_models', 'has_base_model', 'is_finetune', 'is_adapter', 'is_quantized'],
    'Research': ['num_arxiv_refs', 'has_arxiv_ref'],
    'License': ['has_license', 'has_apache_license', 'has_mit_license', 'has_open_license'],
    'Language': ['has_multilingual', 'has_english', 'num_languages'],
    'Dataset': ['num_dataset_refs', 'has_dataset_ref'],
    'Organization': ['is_from_org', 'org_name_length'],
    'Region': ['has_region_us'],
  };

  // Get feature importance map
  const importanceMap = {};
  if (featureImportance) {
    featureImportance.forEach(item => {
      importanceMap[item.feature] = item.importance;
    });
  }

  // Helper to get feature description
  const getFeatureDescription = (key) => {
    const descriptions = {
      'num_tags': 'Total number of tags',
      'days_since_modification': 'Days since last modification',
      'has_transformers': 'Uses Transformers library',
      'has_safetensors': 'Uses SafeTensors format',
      'has_endpoints_compatible': 'Compatible with HuggingFace Endpoints',
      'has_autotrain_compatible': 'Compatible with AutoTrain',
      'num_arxiv_refs': 'Number of arXiv paper references',
      'has_open_license': 'Has open source license (Apache/MIT)',
      'num_languages': 'Number of supported languages',
    };
    return descriptions[key] || '';
  };

  // Sort features by importance if available
  const sortFeatures = (keys) => {
    return keys.sort((a, b) => {
      const impA = importanceMap[a] || 0;
      const impB = importanceMap[b] || 0;
      return impB - impA;
    });
  };

  return (
    <div className="space-y-8">
      {Object.entries(categories).map(([category, keys]) => {
        const availableKeys = keys.filter(key => features.hasOwnProperty(key));
        if (availableKeys.length === 0) return null;

        const sortedKeys = sortFeatures(availableKeys);

        return (
          <div key={category}>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">{category}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {sortedKeys.map(key => (
                <FeatureCard
                  key={key}
                  title={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  value={features[key]}
                  description={getFeatureDescription(key)}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

