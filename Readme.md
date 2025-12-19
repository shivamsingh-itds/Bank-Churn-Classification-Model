# 🏦 Bank Customer Churn Classification

An end-to-end Machine Learning project to predict whether a bank customer will churn (exit) or not, based on demographic, financial, and behavioral features.

This project follows a **modular ML pipeline structure**, including data ingestion, preprocessing, model training, evaluation, and model persistence using `joblib`.

---

## 📌 Problem Statement
Customer churn is a critical issue in the banking sector. Retaining existing customers is often more cost-effective than acquiring new ones.  
The goal of this project is to build a classification model that can accurately predict customer churn and help banks take proactive retention actions.

---

## 📊 Dataset Information
The dataset contains customer information such as:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership
- Estimated Salary
- Churn Status (Target: `Exited`)

Irrelevant identifiers like `RowNumber`, `CustomerId`, and `Surname` were removed during preprocessing.

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA include:
- Churn rate varies significantly across different geographies
- Older customers show a higher tendency to churn
- Inactive members are more likely to exit
- Customers with fewer products churn more frequently
- Class imbalance exists in the target variable

Visualizations include:
- Distribution plots
- Boxplots for numerical features
- Categorical churn comparisons
- Correlation heatmap
- Model accuracy comparison bar chart

---

## ⚙️ Machine Learning Pipeline

### 1. Data Ingestion
- Raw data is loaded from the `data/raw` directory.

### 2. Data Preprocessing
- Dropped unnecessary columns
- Label encoding for categorical features
- Robust scaling for numerical features
- Train-test split
- SMOTE applied **only on training data** to handle class imbalance

### 3. Model Training
Multiple classification models were evaluated.  
**RandomForestClassifier** achieved the highest accuracy and was selected as the final model.

### 4. Model Evaluation
Evaluation metrics used:
- Accuracy
- Classification Report
- Confusion Matrix

> Accuracy alone was not relied upon; class balance and overall performance were considered.

### 5. Model Saving
The trained model is saved using `joblib` for future inference.

---

## 🏆 Best Model Performance
| Model | Accuracy |
|------|----------|
| Random Forest | ~84% |


---

## 🚀 How to Run the Project

1. Clone the repository

```
git clone https://github.com/shivamsingh-itds/Bank-Churn-Classification-Model.git
cd Bank-Churn-Classification-Model
```
2.  Create & activate virtual environment
```
python -m venv ML
ML\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Train the model
```
python -m src.train_pipeline
```

---
## 🧠 Learnings

Importance of proper EDA before modeling

Handling class imbalance using SMOTE

Building modular and reusable ML pipelines

Model persistence using joblib

Accuracy alone is not sufficient for churn problems

---

## 📌 Future Improvements

Hyperparameter tuning (GridSearchCV)

Model deployment using Streamlit or Flask

Recall-focused optimization

Real-time prediction interface

---

## 👤 Author

**Shivam Singh**
Aspiring Data Scientist | Machine Learning Enthusiast

🔗 GitHub: [https://github.com/shivamsingh-itds]
🔗 LinkedIn: [www.linkedin.com/in/shivamsinghds]

---

⭐ If you find this project helpful, feel free to star the repository!