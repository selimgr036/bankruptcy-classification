from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

# SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. SMOTE resampling will not be available.")
    print("Install with: pip install imbalanced-learn")

def get_svm(random_state=42):
    """SVM with RBF kernel"""
    return SVC(kernel='rbf', random_state=random_state, class_weight='balanced', probability=True)

def get_baseline(random_state=42):
    """Baseline that always predicts majority class"""
    return DummyClassifier(strategy='most_frequent', random_state=random_state)

def get_logistic_regression(random_state=42):
    return LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')

def get_random_forest(random_state=42):
    return RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced', n_jobs=-1)

def get_xgboost(random_state=42):
    return xgb.XGBClassifier(random_state=random_state, eval_metric='logloss')

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE for class balancing"""
    if not SMOTE_AVAILABLE:
        return X_train, y_train
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled