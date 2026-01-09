# Bankruptcy & Financial Distress Prediction with Temporal Validation

## Problem Statement

Predicting financial distress and bankruptcy is a critical task for investors, creditors, and financial institutions. Early detection of companies at risk of bankruptcy can help stakeholders make informed decisions and mitigate financial losses. This project aims to develop machine learning models capable of accurately predicting bankruptcy using historical financial data, with a focus on temporal validation to ensure models can generalize to future time periods.

## Data

### Dataset Description

The dataset used in this project is a novel dataset for bankruptcy prediction related to American public companies listed on the New York Stock Exchange (NYSE) and NASDAQ. The dataset comprises:

- **Source**: Accounting data from American public companies
- **Companies**: 8,262 distinct companies
- **Time Period**: 1999-2018
- **Total Observations**: 78,682 firm-year combinations
- **Features**: 18 financial ratios (X1-X18)
- **Target Variable**: Binary classification
  - 0: Company is operating normally (alive)
  - 1: Company filed for bankruptcy (Chapter 11 or Chapter 7)

### Bankruptcy Definition

According to the Security Exchange Commission (SEC), a company in the American market is deemed bankrupt under two circumstances:

1. **Chapter 11 Bankruptcy**: The firm's management files for Chapter 11 of the Bankruptcy Code, indicating an intention to "reorganize" its business. In this case, the company's management continues to oversee day-to-day operations, but significant business decisions necessitate approval from a bankruptcy court.

2. **Chapter 7 Bankruptcy**: The firm's management files for Chapter 7 of the Bankruptcy Code, indicating a complete cessation of operations and the company going out of business entirely.

In this dataset, the fiscal year prior to the filing of bankruptcy under either Chapter 11 or Chapter 7 is labeled as "Bankruptcy" (1) for the subsequent year. Conversely, if the company does not experience these bankruptcy events, it is considered to be operating normally (0).

### Data Characteristics

- **Completeness**: The dataset is complete, without any missing values, synthetic entries, or imputed values
- **Class Distribution**: Imbalanced dataset with 73,462 alive cases (93.4%) and 5,220 bankruptcy cases (6.6%)
- **Temporal Structure**: Data spans 20 years (1999-2018) with firm-year observations

## Models

This project implements four machine learning models:

### 1. Logistic Regression
- **Type**: Linear classifier
- **Rationale**: Provides a baseline model and interpretable coefficients
- **Configuration**: 
  - Class weight balancing to handle imbalanced data
  - Feature scaling required (StandardScaler)
  - `random_state=42` for reproducibility

### 2. Random Forest
- **Type**: Ensemble of decision trees
- **Rationale**: Handles non-linear relationships and provides feature importance
- **Configuration**:
  - 100 trees (n_estimators=100)
  - Class weight balancing
  - `random_state=42` for reproducibility

### 3. XGBoost
- **Type**: Gradient boosting framework
- **Rationale**: Often achieves high performance on structured data
- **Configuration**:
  - Class imbalance handled via `scale_pos_weight`
  - `random_state=42` for reproducibility

### 4. Support Vector Machine (SVM)
- **Type**: Kernel-based classifier (RBF kernel)
- **Rationale**: Can capture complex non-linear patterns
- **Configuration**:
  - RBF kernel
  - Class weight balancing
  - Probability estimates enabled for ROC-AUC calculation
  - Feature scaling required (StandardScaler)
  - `random_state=42` for reproducibility

## Validation Strategy

### Temporal Validation

This project uses **strict temporal validation** (time-based splitting) rather than random train/test splits. This approach is critical for financial prediction tasks because:

1. **Real-world Simulation**: In practice, we predict future bankruptcies based on historical patterns
2. **No Data Leakage**: Prevents information from future time periods from influencing training
3. **Temporal Dependencies**: Financial data often has temporal dependencies that random splits would break

### Split Configuration

The dataset is divided into three subsets based on time periods:

- **Training Set**: 1999-2011 (13 years)
  - Used to train all models
  - Contains historical patterns and relationships

- **Validation Set**: 2012-2014 (3 years)
  - Available for hyperparameter tuning (not used in main pipeline)
  - Could be used for model selection

- **Test Set**: 2015-2018 (4 years)
  - Used exclusively for final model evaluation
  - Represents "future" data that models have never seen
  - Assesses predictive capability in real-world scenarios

### Why Temporal Validation Matters

1. **Economic Cycles**: Financial conditions change over time (recessions, booms)
2. **Regulatory Changes**: Accounting standards and regulations evolve
3. **Market Conditions**: Different economic environments affect bankruptcy patterns
4. **Generalization**: Tests whether models can adapt to changing conditions

## Evaluation Metrics

The project evaluates models using multiple metrics to provide a comprehensive assessment:

1. **Accuracy**: Overall classification correctness
2. **Precision**: Proportion of predicted bankruptcies that are actually bankrupt
3. **Recall**: Proportion of actual bankruptcies that are correctly identified
4. **F1-Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under the Receiver Operating Characteristic curve
6. **PR-AUC**: Area under the Precision-Recall curve (important for imbalanced data)

### Why Multiple Metrics?

- **Imbalanced Data**: Accuracy alone can be misleading with imbalanced classes
- **Business Context**: Different stakeholders care about different metrics (e.g., recall for early warning systems)
- **Comprehensive Assessment**: Multiple metrics provide a fuller picture of model performance

## Model Interpretability

The project includes interpretability analysis using:

1. **SHAP Values**: Shapley Additive Explanations for understanding feature contributions
2. **Feature Importance**: For tree-based models (Random Forest, XGBoost)
3. **Confusion Matrices**: Understanding prediction patterns

## Reproducibility

To ensure reproducibility:

- Fixed random seeds (`random_state=42`) in all models
- Deterministic temporal split boundaries
- Fixed random seed for SHAP computation
- Deterministic data preprocessing pipeline

## Expected Outcomes

1. **Performance Comparison**: Compare four different ML approaches on bankruptcy prediction
2. **Temporal Generalization**: Assess how well models generalize to future time periods
3. **Feature Insights**: Identify which financial ratios are most predictive of bankruptcy
4. **Model Interpretability**: Understand model decisions through SHAP analysis

## Implementation Details

- **Language**: Python 3.9+
- **Key Libraries**: scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn, SHAP
- **Code Structure**: Modular design with separate modules for data loading, modeling, and evaluation
- **Execution**: Single command (`python main.py`) runs complete pipeline

## Future Enhancements

Potential extensions:

1. **Rolling Window Validation**: Test models on multiple time windows
2. **Industry Analysis**: Subgroup analysis by industry sector
3. **Feature Engineering**: Create additional financial ratios
4. **Deep Learning**: Experiment with neural network architectures
5. **Real-time Deployment**: Streamlit dashboard for interactive predictions






