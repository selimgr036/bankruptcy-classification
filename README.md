# Bankruptcy & Financial Distress Prediction with Temporal Validation

A complete machine learning project for predicting financial distress and bankruptcy of American public companies using temporal validation. This project implements a minimal baseline model and four different ML models (Logistic Regression, Random Forest, XGBoost, and SVM) and provides comprehensive evaluation metrics and interpretability analysis. **Success is defined as: models beat the baseline on the temporal test set.**

## Project Overview

This project predicts bankruptcy of firms using machine learning with **strictly time-based validation** (no random splits). The dataset comprises accounting data from 8,262 distinct American public companies listed on NYSE and NASDAQ, recorded during the period spanning from 1999 to 2018.

### Key Features

- **Temporal Validation**: Strict time-based train/validation/test split (train: 1999-2011, validation: 2012-2014, test: 2015-2018)
- **Baseline Model**: Minimal reference (DummyClassifier - always predicts majority class)
- **Four ML Models**: Logistic Regression, Random Forest, XGBoost, and SVM (RBF kernel)
- **Success Criteria**: Models must beat baseline on temporal test set
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- **Baseline Comparison**: Automatic comparison showing which models beat baseline
- **Model Interpretability**: SHAP values and feature importance analysis
- **Reproducibility**: Fixed random seeds (`random_state=42`) throughout

## Dataset

The dataset used in this project is a novel dataset for bankruptcy prediction related to American public companies listed on the New York Stock Exchange and NASDAQ. The dataset comprises:

- **Total observations**: 78,682 firm-year combinations
- **Time period**: 1999-2018
- **Companies**: 8,262 distinct companies
- **Features**: 18 financial ratios (X1-X18)
- **Target**: Binary classification (0 = Alive, 1 = Bankruptcy)

### Bankruptcy Definition

According to the Security Exchange Commission (SEC), a company in the American market is deemed bankrupt under two circumstances:

1. **Chapter 11**: The firm's management files for Chapter 11 of the Bankruptcy Code, indicating an intention to "reorganize" its business.
2. **Chapter 7**: The firm's management files for Chapter 7 of the Bankruptcy Code, indicating a complete cessation of operations.

In this dataset, the fiscal year prior to the filing of bankruptcy under either Chapter 11 or Chapter 7 is labeled as "Bankruptcy" (1) for the subsequent year.

### Temporal Split

The dataset is divided into three subsets based on time periods:

- **Training set**: 1999-2011
- **Validation set**: 2012-2014
- **Test set**: 2015-2018

The test set serves as a means to assess the predictive capability of models in real-world scenarios involving unseen cases.

## Installation

**Important**: This project requires Python 3.11.0. The package versions have been carefully selected for compatibility with Python 3.11.0.

### Using Conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate bankruptcy-prediction
```

If you already have an environment with this name:
```bash
# Remove old environment and recreate
conda env remove -n bankruptcy-prediction
conda env create -f environment.yml

# Or update existing environment
conda env update -n bankruptcy-prediction -f environment.yml --prune
```

## Usage

### Running the Complete Pipeline

Simply run:

```bash
python main.py
```

**Configuration Options:**

You can customize the pipeline using command-line arguments:

```bash
# Default behavior (SMOTE enabled, threshold optimization enabled)
python main.py

# Disable SMOTE for class balancing
python main.py --no-smote

# Disable threshold optimization
python main.py --no-threshold-opt

# Disable both options
python main.py --no-smote --no-threshold-opt
```

**Available Options:**
- `--no-smote`: Disable SMOTE (Synthetic Minority Over-sampling) for class balancing (default: SMOTE is enabled)
- `--no-threshold-opt`: Disable threshold optimization (default: threshold optimization is enabled)

**Features:**
- **Threshold Optimization**: Automatically finds optimal decision thresholds for better precision/recall balance (enabled by default)
- **SMOTE Resampling**: Class balancing using SMOTE (Synthetic Minority Over-sampling) (enabled by default, use `--no-smote` to disable)
- **SVM**: SVM is always included in the model training pipeline
- **Improved Metrics**: Reports both default (0.5) and optimized threshold metrics

This will:

1. Load and preprocess the dataset from `data/raw/american_bankruptcy.csv`
2. Perform temporal train/validation/test split (train: 1999-2011, validation: 2012-2014, test: 2015-2018)
3. Train baseline model (DummyClassifier - majority class) and four ML models (Logistic Regression, Random Forest, XGBoost, SVM)
4. Evaluate all models with comprehensive metrics
5. Compare all models against baseline and report which models beat it (SUCCESS)
6. Generate visualizations (ROC curves, PR curves, confusion matrices, SHAP plots, feature importance)
7. Save all results to `results/` directory

### Output Structure

After running the pipeline, you'll find:

```
results/
├── metrics/
│   ├── model_metrics.csv          # All model metrics in CSV format
│   └── thresholds.json            # Optimized thresholds per model
└── figures/
    ├── roc_curves.png             # ROC curves for all models
    ├── pr_curves.png              # Precision-Recall curves
    ├── confusion_matrices_default.png    # Confusion matrices at default threshold
    ├── confusion_matrices_optimized.png  # Confusion matrices at optimized thresholds
    ├── feature_importance_*.png   # Feature importance (tree models)
    └── shap_summary_*.png         # SHAP summary plots
```

## Project Structure

```
selim-bankruptcy-prediction/
├── README.md                      # This file
├── PROPOSAL.md                    # Project proposal
├── environment.yml                # Conda environment specification
├── main.py                        # Main pipeline script
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── models.py                  # Model definitions
│   ├── evaluation.py              # Evaluation metrics and plotting
│   └── utils.py                   # Utility functions
├── data/
│   └── raw/
│       └── american_bankruptcy.csv # Dataset (must be present)
├── results/
│   ├── metrics/                   # Saved metrics
│   └── figures/                   # Generated plots
└── notebooks/                     # Jupyter notebooks (optional)
```

## Models Implemented

1. **Baseline (DummyClassifier)**: Minimal reference model that always predicts the majority class (most_frequent strategy). This serves as a minimal baseline - any meaningful ML model should outperform this on the temporal test set.

2. **Logistic Regression**: Linear model with class balancing. Previously used as a baseline, now compared against the minimal baseline.

3. **Random Forest**: Ensemble of 100 decision trees with class balancing

4. **XGBoost**: Gradient boosting framework with class imbalance handling

5. **SVM (RBF)**: Support Vector Machine with RBF kernel and class balancing

All models use `random_state=42` for reproducibility and include class balancing to handle the imbalanced dataset (73,462 alive vs 5,220 failed).

### Success Criteria

**Success = Models beat baseline on temporal test set.**

The pipeline automatically compares all models against the baseline and reports:
- Which models beat the baseline (✓ SUCCESS)
- Which models are below the baseline (✗ Below baseline)
- Improvement metrics (absolute and percentage)

The comparison focuses on F1-score with optimized thresholds, but other metrics (accuracy, precision, recall, ROC-AUC, PR-AUC) are also available for comparison.

## Evaluation Metrics

The project computes the following metrics for each model:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve

## Temporal Validation Methodology

This project uses **strict temporal validation** where:

- Models are trained only on historical data (1999-2011)
- Models are validated on intermediate data (2012-2014) for threshold optimization
- Models are tested on future data (2015-2018)
- No random shuffling or data leakage between train, validation, and test sets
- This simulates real-world deployment where we predict future bankruptcies based on past patterns

This approach is crucial for financial prediction tasks where temporal order matters and we want to assess how well models generalize to future time periods.

## Reproducibility

To ensure reproducibility:

- All models use `random_state=42`
- Temporal split boundaries are fixed (train: ≤2011, validation: 2012-2014, test: 2015-2018)
- SHAP computation uses fixed random seed
- Data preprocessing is deterministic

Running `python main.py` multiple times will produce identical results.

## Dependencies

Key dependencies (tested with Python 3.11.0):

- **Python**: 3.11.0
- **pandas**: 2.0.3
- **numpy**: 1.26.4
- **scikit-learn**: 1.4.2
- **xgboost**: 2.0.3
- **matplotlib**: 3.8.4
- **seaborn**: 0.13.2
- **shap**: 0.45.1
- **imbalanced-learn**: 0.11.0
- **tabulate**: 0.9.0

**Version Compatibility Notes**:
- These versions are specifically chosen for compatibility with Python 3.11.0
- pandas 2.0.3 requires numpy < 2.0 (hence numpy 1.26.4)
- imbalanced-learn 0.11.0 requires scikit-learn 1.4.2 (not 1.5+)
- Using newer versions may cause binary incompatibility errors

See `environment.yml` for the complete list with exact versions.

## Troubleshooting

### Binary Incompatibility Errors

If you encounter errors like `ValueError: numpy.dtype size changed`, this indicates a version mismatch. Solutions:

**For conda**: Recreate the environment:
   ```bash
   conda env remove -n bankruptcy-prediction
   conda env create -f environment.yml
   ```

### SMOTE Import Warning

If you see "Warning: imbalanced-learn not installed", check that:
- `imbalanced-learn==0.11.0` is installed
- `scikit-learn==1.4.2` is installed (not 1.5+)
- Both packages are compatible with Python 3.11.0

## Notes

- The dataset is complete with no missing values, so no imputation is needed
- Feature scaling is applied automatically for models that require it (SVM, Logistic Regression)
- **SMOTE/Scaling Order Limitation**: SMOTE is applied to unscaled features before scaling. This is the standard approach (SMOTE should be applied before scaling to avoid scaling synthetic samples), but it means that synthetic samples generated by SMOTE are created in the original feature space and then scaled. This design choice is intentional and follows best practices for handling imbalanced data with SMOTE.
- **SVM Optimization**: 
  - Due to SVM's O(n²) to O(n³) training complexity, the pipeline uses a stratified sample of 20,000 training instances for SVM to significantly speed up training (from hours to minutes)
  - Stratified sampling maintains class balance for better performance
  - **Note**: SVM may have lower performance than tree-based models on this dataset due to the large size and class imbalance
- **SHAP Analysis**: 
  - SHAP values are computed for all models using optimized explainers:
    - **TreeExplainer** for Random Forest and XGBoost (fast and exact)
    - **LinearExplainer** for Logistic Regression (fast and exact)
    - **PermutationExplainer** for SVM (faster than KernelExplainer)
  - For efficiency, SHAP values are computed on a sample of 50 test instances
  - SVM uses a smaller sample (20 instances) due to computational complexity

## License

This project is for educational and research purposes. Please ensure you have the appropriate license to use the bankruptcy dataset.

## Contact

For questions or issues, please refer to the project repository.

