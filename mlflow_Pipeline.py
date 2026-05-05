import pandas as pd
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1. MLflow Experiment Setup
mlflow.set_experiment("IEEE_Fraud_Detection_Assignment")

def run_mlops_pipeline():
    with mlflow.start_run(run_name="XGBoost_SMOTE_Visualized_Run"):
        print("1. Data Ingestion Start ho rahi hai...")
        df = pd.read_csv("train_transaction.csv", nrows=30000)
        
        print("2. Preprocessing & SMOTE Start ho raha hai...")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
                
        cat_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            
        X = df.drop(['isFraud', 'TransactionID'], axis=1)
        y = df['isFraud']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("imbalance_strategy", "SMOTE")
        mlflow.log_param("model", "XGBoost")
        
        print("3. Model Training Start ho rahi hai...")
        scale_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_weight)
        model.fit(X_train_resampled, y_train_resampled)
        
        print("4. Model Evaluation & Metrics Logging...")
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("log_loss", loss)
        
        print("5. Generating & Logging Visualizations...")
        
        # --- Visualization 1: Confusion Matrix ---
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        mlflow.log_figure(fig_cm, "visualizations/confusion_matrix.png")
        plt.close(fig_cm)

        # --- Visualization 2: ROC Curve ---
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        mlflow.log_figure(fig_roc, "visualizations/roc_curve.png")
        plt.close(fig_roc)

        # --- Visualization 3: Top 10 Feature Importances ---
        fig_fi, ax = plt.subplots(figsize=(10, 6))
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        sns.barplot(x=importance, y=importance.index, ax=ax, palette='viridis')
        plt.title('Top 10 Feature Importances')
        mlflow.log_figure(fig_fi, "visualizations/feature_importance.png")
        plt.close(fig_fi)
        
        mlflow.xgboost.log_model(model, "fraud_detection_model")
        print("SUCCESS! Pipeline completed with Visualizations logged to MLflow UI.")

if __name__ == "__main__":
    run_mlops_pipeline()