from sklearn.model_selection import train_test_split
import pandas as pd

from src.data_preprocessing import load_data, clean_data, encode_data, split_features_target
from src.model import train_random_forest, train_xgboost, predict
from src.evaluate import evaluate_model
from src.utils import risk_category

df = load_data("data/churn.csv")

df = clean_data(df)
df = encode_data(df)

X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = train_random_forest(X_train, y_train)
xgb = train_xgboost(X_train, y_train)

rf_metrics = evaluate_model(y_test, rf.predict(X_test))
xgb_metrics = evaluate_model(y_test, xgb.predict(X_test))

print("Random Forest:", rf_metrics)
print("XGBoost:", xgb_metrics)

preds, probs = predict(xgb, X)

df["Churn_Prediction"] = preds
df["Risk_Score"] = probs
df["Risk_Category"] = df["Risk_Score"].apply(risk_category)

important_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Credit card (automatic)",
    "Churn_Yes",
    "Churn_Prediction",
    "Risk_Score",
    "Risk_Category"
]

df_final = df[important_cols]
df_final.to_csv("churn_predictions.csv", index=False)

print("Project Successfull")