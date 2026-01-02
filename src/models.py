from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

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