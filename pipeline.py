from kfp import dsl
from kfp import compiler

# --- COMPONENT 1: Data Ingestion (REAL IEEE DATA) ---
@dsl.component(packages_to_install=['pandas', 'requests'])
def ingest_data(output_csv: dsl.OutputPath('CSV')):
    import pandas as pd
    
    print("Downloading dataset from local host...")
    # host.minikube.internal is how Minikube talks to your Windows machine
    url = "http://host.minikube.internal:8000/train_transaction.csv"
    
    # We use nrows=30000 to prevent your 7GB cluster from crashing with Out-Of-Memory errors
    df = pd.read_csv(url, nrows=30000)
    
    # Save to Kubeflow artifact storage
    df.to_csv(output_csv, index=False)
    print(f"Data Ingestion Complete. Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# --- COMPONENT 2: Preprocessing, Encoding & SMOTE ---
@dsl.component(packages_to_install=['pandas', 'scikit-learn', 'imbalanced-learn'])
def preprocess_data(input_csv: dsl.InputPath('CSV'), 
                    X_train_path: dsl.OutputPath('CSV'), 
                    X_test_path: dsl.OutputPath('CSV'), 
                    y_train_path: dsl.OutputPath('CSV'), 
                    y_test_path: dsl.OutputPath('CSV')):
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE
    
    df = pd.read_csv(input_csv)
    
    # Task 2: Handle Missing Values (Advanced Strategy - Median for numerical, Mode for categorical)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
            
    # Task 2: High-Cardinality Categorical Features (Label Encoding)
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    X = df.drop(['isFraud', 'TransactionID'], axis=1) # isFraud is the target
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Task 2: Class Imbalance Strategy - SMOTE
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    X_train_resampled.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train_resampled.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print("Preprocessing and SMOTE complete.")

# --- COMPONENT 3: Model Training (XGBoost) ---
@dsl.component(packages_to_install=['pandas', 'xgboost', 'scikit-learn'])
def train_model(X_train_path: dsl.InputPath('CSV'), 
                y_train_path: dsl.InputPath('CSV'),
                model_output: dsl.OutputPath('Model')):
    import pandas as pd
    from xgboost import XGBClassifier
    import pickle
    
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    
    # Task 4: Cost-Sensitive Learning (Higher penalty for false negatives)
    scale_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_weight)
    model.fit(X_train, y_train)
    
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    print("XGBoost Model Trained successfully.")

# --- PIPELINE DEFINITION ---
@dsl.pipeline(name="fraud-detection-pipeline", description="IEEE Fraud Detection Pipeline")
def fraud_pipeline():
    ingest_task = ingest_data()
    
    preprocess_task = preprocess_data(
        input_csv=ingest_task.outputs['output_csv']
    )
    
    train_task = train_model(
        X_train_path=preprocess_task.outputs['X_train_path'],
        y_train_path=preprocess_task.outputs['y_train_path']
    )

# --- EXECUTION & TRACKER ---
if __name__ == '__main__':
    print("Starting pipeline compilation...")
    try:
        compiler.Compiler().compile(pipeline_func=fraud_pipeline, package_path='fraud_pipeline.yaml')
        print("SUCCESS: 'fraud_pipeline.yaml' has been generated successfully! Please check your directory.")
    except Exception as e:
        print("ERROR ENCOUNTERED! Details:")
        print(e)