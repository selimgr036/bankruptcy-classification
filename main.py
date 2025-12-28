import numpy as np
from src.data_loader import load_data, preprocess_data, split_by_year
from src.models import get_logistic_regression, get_random_forest
from src.evaluation import evaluate_model
from sklearn.preprocessing import StandardScaler
from src.models import get_logistic_regression, get_random_forest, get_xgboost
from src.evaluation import evaluate_model, plot_roc_curves
from pathlib import Path

# Load and split
df = load_data()
train_df, test_df = split_by_year(df)

X_train, y_train, _ = preprocess_data(train_df)
X_test, y_test, _ = preprocess_data(test_df)

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
models = {
    'Logistic Regression': (get_logistic_regression(), X_train_scaled, X_test_scaled),
    'Random Forest': (get_random_forest(), X_train, X_test),
    'XGBoost': (get_xgboost(), X_train, X_test)
}

for name, (model, X_tr, X_te) in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_tr, y_train)
    metrics, _ = evaluate_model(model, X_te, y_test)
    print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

# Store results for plotting
results_for_plot = {}
for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train)
    results_for_plot[name] = (model, X_te, y_test)

# Plot ROC curves
Path("results/figures").mkdir(parents=True, exist_ok=True)
plot_roc_curves(results_for_plot, "results/figures/roc_curves.png")
print("\nSaved ROC curves to results/figures/roc_curves.png")