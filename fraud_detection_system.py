import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data Processing Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# ML Models - Supervised Learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ML Models - Unsupervised Learning
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Save Models
import pickle

print("=" * 60)
print("FINANCIAL FRAUD DETECTION SYSTEM")
print("=" * 60)

# STEP 1: DATA LOADING & EXPLORATION

df = pd.read_csv('financial_fraud_dataset.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData Info:")
print(df.info())

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nFraud Distribution:")
print(df['is_fraud'].value_counts())
print(f"Fraud Percentage: {(df['is_fraud'].sum() / len(df)) * 100:.2f}%")

# STEP 2: DATA PREPROCESSING


# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract temporal features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print(" Temporal features extracted")

# Label Encoding for categorical variables
le_merchant = LabelEncoder()
le_location = LabelEncoder()
le_device = LabelEncoder()

df['merchant_category_encoded'] = le_merchant.fit_transform(df['merchant_category'])
df['customer_location_encoded'] = le_location.fit_transform(df['customer_location'])
df['device_type_encoded'] = le_device.fit_transform(df['device_type'])

print(" Categorical variables encoded")

# Select features for modeling
feature_columns = ['amount', 'customer_age', 'previous_transactions', 
                   'merchant_category_encoded', 'customer_location_encoded', 
                   'device_type_encoded', 'hour', 'day_of_week', 'month', 'is_weekend']

X = df[feature_columns]
y = df['is_fraud']

print(f" Feature matrix shape: {X.shape}")
print(f" Target variable shape: {y.shape}")

# STEP 3: ETL - SAVE TO DATABASE


conn = sqlite3.connect("fraud_detection.db")
cursor = conn.cursor()

# Create transactions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    timestamp TEXT,
    amount REAL,
    merchant_category TEXT,
    customer_id TEXT,
    customer_age INTEGER,
    customer_location TEXT,
    device_type TEXT,
    previous_transactions INTEGER,
    is_fraud INTEGER
)
""")

# Insert data
df.to_sql("transactions", conn, if_exists="replace", index=False)
conn.commit()
conn.close()

print(" Data saved to fraud_detection.db")

# STEP 4: TRAIN-TEST SPLIT & SCALING


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# STEP 5: MODEL TRAINING - SUPERVISED LEARNING

print("-" * 60)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

trained_models = {}
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC Score: {roc_auc:.4f}")
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'ROC-AUC': roc_auc
    })
    
    trained_models[name] = model
    
    # Detailed metrics for best model
    if name == 'Random Forest':
        print(f"\n{name} - Detailed Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

# Display Results Summary
print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Select best model (Random Forest)
best_model = trained_models['Random Forest']
print(f"\n Best Model Selected: Random Forest")

# STEP 6: UNSUPERVISED LEARNING - ANOMALY DETECTION


# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_predictions = iso_forest.fit_predict(X_train_scaled)
iso_anomalies = np.sum(iso_predictions == -1)

print(f"Isolation Forest - Anomalies Detected: {iso_anomalies}")

# One-Class SVM (on smaller sample for speed)
sample_size = min(5000, len(X_train_scaled))
X_sample = X_train_scaled[:sample_size]

oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
oc_svm.fit(X_sample)
svm_predictions = oc_svm.predict(X_test_scaled)
svm_anomalies = np.sum(svm_predictions == -1)

print(f"One-Class SVM - Anomalies Detected: {svm_anomalies}")

# STEP 7: FEATURE IMPORTANCE


feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head())

# STEP 8: SAVE MODELS


# Save Random Forest model
with open('fraud_detection_rf_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Random Forest model saved: fraud_detection_rf_model.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved: scaler.pkl")

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'merchant': le_merchant,
        'location': le_location,
        'device': le_device
    }, f)
print("Label encoders saved: label_encoders.pkl")

# STEP 9: FRAUD PREDICTION FUNCTION


def predict_fraud(transaction_data):
    """
    Predict fraud for new transaction
    
    Parameters:
    transaction_data: dict with keys:
        - amount: float
        - merchant_category: str
        - customer_age: int
        - customer_location: str
        - device_type: str
        - previous_transactions: int
        - timestamp: str (format: 'YYYY-MM-DD HH:MM:SS')
    
    Returns:
    dict with prediction and probability
    """
    # Load models
    with open('fraud_detection_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    # Extract temporal features
    ts = pd.to_datetime(transaction_data['timestamp'])
    hour = ts.hour
    day_of_week = ts.dayofweek
    month = ts.month
    is_weekend = 1 if day_of_week in [5, 6] else 0
    
    # Encode categorical variables
    merchant_encoded = encoders['merchant'].transform([transaction_data['merchant_category']])[0]
    location_encoded = encoders['location'].transform([transaction_data['customer_location']])[0]
    device_encoded = encoders['device'].transform([transaction_data['device_type']])[0]
    
    # Create feature array
    features = np.array([[
        transaction_data['amount'],
        transaction_data['customer_age'],
        transaction_data['previous_transactions'],
        merchant_encoded,
        location_encoded,
        device_encoded,
        hour,
        day_of_week,
        month,
        is_weekend
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return {
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability),
        'risk_level': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'
    }

print("Prediction function created")

# STEP 10: TEST PREDICTION

# Example transaction
test_transaction = {
    'amount': 500.0,
    'merchant_category': 'electronics',
    'customer_age': 35,
    'customer_location': 'NY',
    'device_type': 'mobile',
    'previous_transactions': 5,
    'timestamp': '2024-10-28 14:30:00'
}

prediction_result = predict_fraud(test_transaction)

print("\nTest Transaction:")
print(f"  Amount: ${test_transaction['amount']}")
print(f"  Category: {test_transaction['merchant_category']}")
print(f"  Location: {test_transaction['customer_location']}")
print(f"\nPrediction Result:")
print(f"  Fraud: {prediction_result['is_fraud']}")
print(f"  Probability: {prediction_result['fraud_probability']:.4f}")
print(f"  Risk Level: {prediction_result['risk_level']}")

# STEP 11: VISUALIZATION

# Create visualization directory
import os
os.makedirs('fraud_visualizations', exist_ok=True)

# 1. Fraud Distribution
plt.figure(figsize=(8, 6))
df['is_fraud'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Fraud vs Non-Fraud Transactions', fontsize=14, fontweight='bold')
plt.xlabel('Is Fraud (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fraud_visualizations/fraud_distribution.png')
#print("Saved: fraud_distribution.png")
plt.close()

# 2. Feature Importance
plt.figure(figsize=(10, 6))
feature_importance.plot(x='Feature', y='Importance', kind='barh', color='steelblue', legend=False)
plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('fraud_visualizations/feature_importance.png')
#print("Saved: feature_importance.png")
plt.close()

# 3. Model Performance Comparison
plt.figure(figsize=(10, 6))
results_df.set_index('Model')[['Accuracy', 'ROC-AUC']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xlabel('Models')
plt.xticks(rotation=45, ha='right')
plt.legend(['Accuracy', 'ROC-AUC'])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('fraud_visualizations/model_comparison.png')
#print(" Saved: model_comparison.png")
plt.close()

