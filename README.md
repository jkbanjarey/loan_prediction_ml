# 🏦 Loan Prediction Model  

## 📌 Overview  
This project aims to predict **loan approvals** based on applicant details using **Machine Learning** techniques. The model helps financial institutions assess loan eligibility efficiently.  

## ✨ Features  
- 📊 **Predicts loan approval based on applicant details.**  
- 🔍 **Handles missing values and categorical data effectively.**  
- 🤖 **Utilizes machine learning models like Logistic Regression, Decision Trees, and Random Forest.**  
- 📈 **Evaluates model performance using accuracy and F1-score.**  

## 📂 Dataset  
The dataset **LoanApprovalPrediction.csv** contains:  
- **Applicant Information:** Gender, Married, Dependents, Education, Employment status.  
- **Financial Details:** Income, Loan amount, Credit history, Loan term.  
- **Target Variable:** Loan_Status (1 = Approved, 0 = Rejected).  

### 🔄 Data Preprocessing  
- **Checked for missing values and handled them using median/mode imputation.**  
- **Encoded categorical features using Label Encoding.**  
- **Applied feature scaling (StandardScaler) to normalize numerical values.**  

## 🛠 Requirements  
Install the necessary dependencies:  
```bash  
pip install numpy pandas matplotlib seaborn scikit-learn  
```  

## 🏗 Model Architecture  
The following models were tested:  
1. **Logistic Regression** – A simple yet effective baseline classifier.  
2. **Decision Tree** – Captures non-linearity in loan approval patterns.  
3. **Random Forest** – An ensemble method for improved accuracy.  
4. **XGBoost (Optional)** – Used for optimizing performance.  

## 🏋️‍♂️ Training & Evaluation  
- **Split the dataset into training (80%) and testing (20%).**  
- **Trained models and compared their accuracy.**  
- **Evaluated models using Precision, Recall, F1-score, and Confusion Matrix.**  

## 📊 Insights & Results  
### 🔍 Key Insights:  
- **Applicants with a higher income and strong credit history have higher approval rates.**  
- **Loan amounts above a certain threshold decrease approval chances.**  
- **Marital status and education level impact approval probability.**  

### 📈 Model Performance:  
| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|-----------|-----------|--------|----------|
| Logistic Regression | 78.5%     | 75.2%     | 79.1%  | 77.1%    |
| Decision Tree      | 72.8%     | 70.1%     | 74.3%  | 72.1%    |
| Random Forest     | **81.2%**  | **78.9%**  | **82.4%** | **80.6%** |

- **Random Forest performed the best, achieving the highest accuracy and F1-score.**  
- **Feature importance analysis** revealed that Credit History, Income, and Loan Amount were the top factors.  

## 🚀 Usage  
To run the project, execute:  
```bash  
jupyter notebook loan_prediction_ml.ipynb  
```  

## 🔮 Future Enhancements  
- 🔄 **Improve feature engineering with additional financial data.**  
- 🤖 **Test deep learning models (ANNs) for better performance.**  
- 🏦 **Deploy as an API for real-time loan approvals.**  

## 👨‍💻 Author  
**Jitendra Kumar Banjarey**  

## 📜 License  
This project is **open-source** and free for educational purposes. 🎓  
