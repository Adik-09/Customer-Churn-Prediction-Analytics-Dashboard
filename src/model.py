from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs