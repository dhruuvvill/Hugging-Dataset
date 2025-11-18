# Testing the Popularity Prediction Model

## Quick Start

### Option 1: Run Full Training (Recommended)
```bash
python model.py
```
This will train:
- Regression model for downloads
- Classification model for high/low popularity (downloads)
- Regression model for likes

### Option 2: Run Quick Test
```bash
python test_model.py
```
This will train only the classification model (faster).

## What to Expect

The model will:
1. **Load data** from `hf_features/index_models.csv`
2. **Extract features** from tags, dates, and descriptions
3. **Train** a Random Forest model
4. **Print metrics**:
   - For regression: RMSE and R² score
   - For classification: Accuracy, Precision, Recall, F1
5. **Show feature importance** (top 10 features)
6. **Save the trained model** to `models_popularity/` directory

## Sample Output

```
============================================================
Loading data from: huggingface_metadata/datasets_metadata.csv
============================================================

Loaded 12345 datasets with downloads data
Target distribution:
count    12345.00
mean      45678.90
std      123456.78
min          10.00
max     2000000.00

Extracting tag features...
Extracting date features...
Extracting description features...
Training set: 9876 samples
Test set: 2469 samples

============================================================
Regression Results:
============================================================
RMSE: 12345.67
R² Score: 0.6543

Top 10 Most Important Features:
num_tags                                  : 0.1234
days_since_modification                   : 0.0987
has_task_category                         : 0.0876
author_microsoft                          : 0.0754
main_task_text-classification             : 0.0632
has_library_datasets                      : 0.0543
description_length                        : 0.0432
is_recent                                 : 0.0321
license_type_mit                          : 0.0210
has_link                                  : 0.0156

Model saved to: models_popularity/popularity_regression_downloads.joblib
```

## Troubleshooting

### Error: "Module not found: sklearn"
```bash
pip install -r requirement.txt
```

### Error: "File not found"
Make sure you're in the project directory:
```bash
cd /Users/dhruvil1904/Desktop/HuggingFace
```

### Low R² Score
This is normal! Popularity prediction is challenging. Try:
- Increasing model complexity
- Adding more features
- Using ensemble methods

## Next Steps

After training:
1. Check the saved models in `models_popularity/`
2. Load a model to make predictions on new data
3. Analyze feature importance to understand what makes datasets popular

## Load and Use Trained Model

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('models_popularity/popularity_classification_downloads.joblib')

# Load new data
new_data = pd.read_csv('new_datasets.csv')

# Prepare features (same preprocessing as training)
from model import prepare_features_advanced
X = prepare_features_advanced(new_data)

# Make predictions
predictions = model.predict(X)
```

## Understanding the Model

The model predicts popularity based on:
- **Tag features**: Task categories, languages, licenses, modality
- **Date features**: How recently the dataset was updated
- **Description features**: Length, presence of code examples/links
- **Author features**: Who created the dataset

Feature importance shows which factors matter most for popularity!

