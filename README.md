# Financial Fraud Detection System

A comprehensive machine learning system for detecting financial fraud using supervised and unsupervised learning techniques. This project implements multiple classification models and anomaly detection algorithms to identify fraudulent transactions with high accuracy.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results & Visualizations](#results--visualizations)
- [Key Features & Importance](#key-features--importance)
- [Contributing](#contributing)
- [Collaborators](#collaborators)

## Project Overview

This project builds a robust fraud detection system that processes financial transaction data, trains multiple machine learning models, and deploys them for real-time fraud prediction. The system combines both supervised learning (classification) and unsupervised learning (anomaly detection) approaches to maximize fraud detection capabilities.

### Problem Statement

Financial fraud is a critical issue for financial institutions. This system aims to:
- Identify fraudulent transactions with high accuracy
- Minimize false positives to reduce customer friction
- Provide real-time risk assessment for transactions
- Enable quick deployment through pickled model artifacts

## Features

### Data Processing
- **Temporal Feature Extraction**: Hour, day of week, month, weekend indicator from timestamps
- **Label Encoding**: Categorical variables encoding (merchant category, location, device type)
- **Feature Scaling**: StandardScaler normalization for ML models
- **Class Balancing**: Stratified train-test split to handle imbalanced fraud distribution

### Supervised Learning Models
- **Logistic Regression**: Baseline linear classifier
- **Decision Tree**: Tree-based single model
- **Random Forest**: Ensemble method (best performer)
- **Gradient Boosting**: Sequential ensemble learning
- **XGBoost**: Extreme gradient boosting classifier

### Unsupervised Learning
- **Isolation Forest**: Anomaly detection via random isolation
- **One-Class SVM**: Support vector machine for outlier detection

### Database Integration
- **SQLite Integration**: Persistent transaction storage
- **ETL Pipeline**: Data extraction, transformation, and loading to database

### Model Persistence
- Trained model serialization using pickle
- Scaler and encoder serialization for deployment
- Ready-to-use prediction functions

## Technologies Used

```
Core Libraries:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Machine learning models and preprocessing
- xgboost: Gradient boosting classifier
- sqlite3: Database management

Visualization:
- matplotlib: Static plotting
- seaborn: Statistical data visualization

Model Deployment:
- pickle: Model serialization
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/Ushasri-kolla/FRUAD_DETECTION.git
cd FRUAD_DETECTION

# Install required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# Run the fraud detection system
python fraud_detection_system.py
```

## Project Structure

```
FRUAD_DETECTION/
├── fraud_detection_system.py           # Main system implementation
├── financial_fraud_dataset.csv         # Transaction dataset
├── fraud_detection.db                  # SQLite database
├── fraud_detection_rf_model.pkl        # Trained Random Forest model
├── scaler.pkl                          # Feature scaling object
├── label_encoders.pkl                  # Categorical encoders
├── fraud_distribution.png              # Fraud distribution visualization
├── feature_importance.png              # Feature importance chart
├── model_comparison.png                # Model performance comparison
└── Financial Fraud Detection Project.txt  # Project documentation
```

## Usage

### Running the Complete System

```bash
python fraud_detection_system.py
```

This will:
1. Load and explore the financial fraud dataset
2. Preprocess data with temporal and categorical features
3. Save data to SQLite database
4. Train multiple ML models
5. Evaluate model performance
6. Detect anomalies using unsupervised learning
7. Generate visualizations
8. Save trained models for deployment

### Making Predictions on New Transactions

```python
from fraud_detection_system import predict_fraud

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

# Get prediction
result = predict_fraud(test_transaction)
print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.4f}")
print(f"Risk Level: {result['risk_level']}")
```

**Output:**
```
Fraud: False
Probability: 0.1234
Risk Level: LOW
```

### Risk Levels
- **LOW**: Fraud probability < 0.3
- **MEDIUM**: Fraud probability 0.3 - 0.7
- **HIGH**: Fraud probability > 0.7

## Model Performance

The system trains and compares multiple models:

| Model | Accuracy | ROC-AUC Score |
|-------|----------|---------------|
| Logistic Regression | ~94% | ~0.92 |
| Decision Tree | ~95% | ~0.94 |
| Random Forest | ~97% | ~0.96 |
| Gradient Boosting | ~96% | ~0.95 |
| XGBoost | ~96% | ~0.95 |

**Best Model**: Random Forest (97% accuracy, 0.96 ROC-AUC)

### Classification Report (Random Forest)
- **Precision**: High - Minimizes false positives
- **Recall**: High - Catches most fraudulent transactions
- **F1-Score**: Balanced performance across classes

## Results & Visualizations

### 1. Fraud Distribution
Shows the balance between fraudulent and non-fraudulent transactions in the dataset.

### 2. Feature Importance
Top contributing features for fraud detection:
- Transaction amount
- Customer age
- Previous transaction history
- Merchant category
- Device type
- Location information
- Temporal patterns (hour, day of week)

### 3. Model Comparison
Comparative analysis of all models showing accuracy and ROC-AUC scores.

## Key Features & Importance

The system identifies these as most important for fraud detection:

1. **Transaction Amount**: Large unusual amounts trigger alerts
2. **Customer History**: Previous transaction count indicates patterns
3. **Temporal Patterns**: Hour of transaction, weekend vs weekday
4. **Location**: Unusual geographic patterns
5. **Device Type**: Device consistency checks
6. **Merchant Category**: High-risk merchant categories

## Anomaly Detection Results

### Isolation Forest
- Successfully identifies ~10% of transactions as anomalies
- Useful for detecting novel fraud patterns

### One-Class SVM
- Complements supervised learning approaches
- Catches outliers using support vector techniques

## Database Schema

### Transactions Table
```sql
CREATE TABLE transactions (
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
);
```

## Model Serialization Files

- **fraud_detection_rf_model.pkl**: Trained Random Forest classifier
- **scaler.pkl**: StandardScaler for feature normalization
- **label_encoders.pkl**: Dictionary of LabelEncoders for categorical variables

## Performance Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **ROC-AUC Score**: Measures ability to distinguish between fraud/non-fraud across thresholds
- **Precision**: Of predicted frauds, how many are actually fraudulent
- **Recall**: Of actual frauds, how many we correctly identify
- **F1-Score**: Harmonic mean of precision and recall

## Future Enhancements

- [ ] Deep learning models (LSTM, Neural Networks)
- [ ] Real-time API deployment
- [ ] Model interpretability with SHAP values
- [ ] Handling concept drift with periodic retraining
- [ ] Imbalanced learning techniques (SMOTE)
- [ ] Ensemble voting systems
- [ ] Web dashboard for monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Collaborators

**Chetan29-30** (Chetankumar Ganesh Mete)
- GitHub: [@Chetan29-30](https://github.com/Chetan29-30)
- Role: Co-Developer
- Contributions: README documentation, project setup, and ML pipeline architecture



## Author

**Ushasri Kolla**
- GitHub: [@Ushasri-kolla](https://github.com/Ushasri-kolla)

## Acknowledgments

- scikit-learn for robust ML algorithms
- XGBoost for powerful gradient boosting
- The financial fraud detection community for best practices

---

**Note**: This system is for educational and research purposes. Always validate with domain experts before deployment in production environments.
