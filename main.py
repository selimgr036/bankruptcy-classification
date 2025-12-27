import numpy as np
from src.data_loader import load_data, preprocess_data, split_by_year
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and split
df = load_data()
train_df, test_df = split_by_year(df)

X_train, y_train, _ = preprocess_data(train_df)
X_test, y_test, _ = preprocess_data(test_df)

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
print(f"LR - Accuracy: {accuracy_score(y_test, lr_pred):.4f}, F1: {f1_score(y_test, lr_pred):.4f}")

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)  # RF doesn't need scaling
rf_pred = rf_model.predict(X_test)
print(f"RF - Accuracy: {accuracy_score(y_test, rf_pred):.4f}, F1: {f1_score(y_test, rf_pred):.4f}")