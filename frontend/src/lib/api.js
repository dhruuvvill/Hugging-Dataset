const API_BASE_URL = 'http://localhost:8000';

export const predictModel = async (url) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        huggingface_url: url
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Prediction failed');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return await response.json();
  } catch (error) {
    console.error('Health check error:', error);
    return { status: 'error' };
  }
};

export const extractFeatures = async (url) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/extract-features`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        huggingface_url: url
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Feature extraction failed');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Feature extraction error:', error);
    throw error;
  }
};

export const extractFeaturesById = async (modelId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/extract-features/${modelId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Feature extraction failed');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Feature extraction error:', error);
    throw error;
  }
};

export const getTrendingDatasets = async (limit = 10) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/datasets/trending?limit=${limit}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch trending datasets');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Trending datasets error:', error);
    throw error;
  }
};

export const getTopDatasets = async (limit = 10, sortBy = 'downloads') => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/datasets/top?limit=${limit}&sort_by=${sortBy}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch top datasets');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Top datasets error:', error);
    throw error;
  }
};

export const getDatasetStats = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/datasets/stats`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch dataset statistics');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Dataset stats error:', error);
    throw error;
  }
};

