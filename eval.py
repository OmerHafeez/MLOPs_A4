import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import shap
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 1. Setup MLflow for tracking everything
mlflow.set_experiment("IEEE_Fraud_Advanced_Analysis")

def run_advanced_pipeline():
    with mlflow.start_run(run_name="LightGBM_v_XGBoost_Comparison"):
        # --- Task 1: Data Load ---
        df = pd.read_csv("train_transaction.csv", nrows=30000)
        
        # --- Task 2: Advanced Imputation & Encoding ---
        X = df.drop(['isFraud', 'TransactionID'], axis=1)
        y = df['isFraud']
        
        # Simple mode/median imputation
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode()[0])
                X[col] = pd.factorize(X[col])[0] # Fast Label Encoding
            else:
                X[col] = X[col].fillna(X[col].median())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # --- Task 2: SMOTE Strategy ---
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        # --- Task 3 & 4: LightGBM with Cost-Sensitive Learning ---
        # Assignment Requirement: Higher penalty for False Negatives
        # we use 'isFraud' ratio to set scale_pos_weight
        fraud_ratio = (y_res == 0).sum() / (y_res == 1).sum()
        
        print("Training LightGBM (The Modern Brain)...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            scale_pos_weight=fraud_ratio * 1.5, # Extra penalty for missing fraud!
            learning_rate=0.05
        )
        lgb_model.fit(X_res, y_res)

        # --- Task 3: Evaluation (Full Metrics) ---
        preds = lgb_model.predict(X_test)
        probs = lgb_model.predict_proba(X_test)[:, 1]

        metrics = {
            "Recall (Fraud Catch Rate)": recall_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "F1_Score": f1_score(y_test, preds),
            "AUC_ROC": roc_auc_score(y_test, probs)
        }

        # Log metrics to MLflow
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"{name}: {value:.4f}")

        # --- Task 9: Explainability (SHAP) ---
        print("Calculating SHAP values (Why is it fraud?)...")
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_test.iloc[:100])
        
        # Save SHAP plot
        plt.figure()
        shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        
        # Save Model
        mlflow.lightgbm.log_model(lgb_model, "lightgbm_fraud_model")
        
        print("SUCCESS: Advanced metrics and Explainability logged!")

if __name__ == "__main__":
    run_advanced_pipeline()