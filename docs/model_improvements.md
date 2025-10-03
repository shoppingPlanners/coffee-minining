# Model Improvements for Coffee Health Risk Classification

## Problem Identified

The original classification model had several critical issues:

1. **Severe Class Imbalance**:
   - Mild: ~88% (3579 samples)
   - Moderate: ~11% (463 samples)  
   - Severe: ~0.4% (17 samples)

2. **Poor Performance on Minority Classes**:
   - Logistic Regression: Only 47% recall on class 1, 33% recall on class 2
   - Models were overfitting to the majority class
   - High overall accuracy (96-99%) but poor real-world performance

3. **Inappropriate Evaluation Metrics**:
   - Focus on accuracy rather than balanced metrics
   - No consideration of per-class performance
   - Missing F1-scores and balanced accuracy

4. **Lack of Proper Handling Techniques**:
   - No oversampling/undersampling
   - No class weights
   - Basic hyperparameters without tuning

## Solutions Implemented

### 1. SMOTE (Synthetic Minority Over-sampling Technique)
- Applied SMOTE to balance the training data
- Generates synthetic examples of minority classes
- Balances all classes to equal representation during training

### 2. Class Weighting
- Added `class_weight='balanced'` to all compatible models
- Automatically adjusts weights inversely proportional to class frequencies
- Helps models pay more attention to minority classes

### 3. Improved Model Hyperparameters
- **Random Forest**:
  - Increased n_estimators to 300
  - Set max_depth=15 to prevent overfitting
  - Added min_samples_split=10 and min_samples_leaf=4
  - Used class_weight='balanced'

- **Gradient Boosting**:
  - n_estimators=200 with learning_rate=0.05
  - max_depth=7 for better generalization
  - min_samples_split=10 to prevent overfitting

- **XGBoost**:
  - Added scale_pos_weight parameter
  - Reduced learning_rate to 0.05
  - Tuned tree depth and child weight

### 4. Comprehensive Evaluation Metrics
- **Balanced Accuracy**: Average recall per class
- **F1-Score (Macro)**: Unweighted mean of F1 per class
- **F1-Score (Weighted)**: Weighted mean by class support
- **Per-class Precision, Recall, F1-Score**
- **Confusion Matrix (both raw and normalized)**

### 5. Hyperparameter Tuning
- GridSearchCV with StratifiedKFold (k=3)
- Optimizing for F1-macro score (better for imbalanced data)
- Systematic search over parameter space

## Expected Improvements

### Before Improvements:
```
Logistic Regression:
  Class 0 (Mild): 97% precision, 99% recall
  Class 1 (Moderate): 69% precision, 47% recall
  Class 2 (Severe): 100% precision, 33% recall
```

### After Improvements:
- Better recall on minority classes (expected 70-90%)
- More balanced performance across all classes
- F1-macro score as primary metric (expected 0.75-0.90)
- Reduced overfitting through regularization

## Key Files Updated

1. **notebooks/05_model_building.ipynb**: 
   - Complete overhaul with SMOTE
   - Multiple evaluation metrics
   - Hyperparameter tuning

2. **src/models.py**:
   - Added `apply_smote()` function
   - Improved training functions with better defaults
   - Enhanced evaluation with balanced metrics

3. **requirements.txt**:
   - Added `imbalanced-learn` package

## How to Use

### Training a New Model:
```python
from src.models import apply_smote, train_random_forest, evaluate_model

# Apply SMOTE
X_train_balanced, y_train_balanced, smote = apply_smote(X_train, y_train)

# Train model
model = train_random_forest(X_train_balanced, y_train_balanced)

# Evaluate
y_pred, metrics = evaluate_model(model, X_test, y_test)
print(f"F1-Macro: {metrics['f1_macro']:.4f}")
```

### Running the Pipeline:
```bash
cd notebooks
jupyter notebook 05_model_building.ipynb
```

Or run all notebooks:
```bash
python scripts/pipeline.py --run-all
```

## Best Practices for Imbalanced Classification

1. ✅ **Always check class distribution** before training
2. ✅ **Use stratified splitting** to maintain class ratios
3. ✅ **Apply SMOTE or class weights** for imbalanced data
4. ✅ **Evaluate with F1-score and balanced accuracy** instead of just accuracy
5. ✅ **Monitor per-class metrics** to ensure all classes perform well
6. ✅ **Use cross-validation with stratification** for reliable estimates
7. ✅ **Regularize models** to prevent overfitting to majority class

## Next Steps

1. Run the improved model and compare results
2. Consider ensemble methods if single model performance is insufficient
3. Collect more data for minority classes if possible
4. Experiment with cost-sensitive learning
5. Try other resampling techniques (SMOTETomek, ADASYN)


