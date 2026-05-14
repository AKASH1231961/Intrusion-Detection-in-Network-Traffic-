# Intrusion Detection System Using Machine Learning

## 📌 Project Overview
This project focuses on building an **Intrusion Detection System (IDS)** using Machine Learning techniques to classify network traffic as either **normal** or **malicious**.

The system analyzes network connection data from the **KDD Cup 1999 dataset** and applies data preprocessing, feature engineering, feature selection, and multiple machine learning models for attack detection.

The project aims to improve cybersecurity monitoring by identifying suspicious network activities efficiently and accurately.

---

# 👥 Team Members
- Member 1 – AKASH S
- Member 2 – EVANS
- Member 3 – LANIYA MOHAN
  




---

# ❗ Problem Statement
Cyberattacks such as Denial of Service (DoS), unauthorized access, and probing attacks pose serious threats to modern computer networks. Traditional rule-based security systems struggle to detect new and evolving attack patterns.

This project aims to develop a Machine Learning-based Intrusion Detection System capable of:
- Detecting malicious network traffic
- Classifying different attack categories
- Improving network security monitoring

---

# 💡 Motivation
With the rapid growth of internet usage and cloud-based services, protecting network infrastructure has become essential. Machine Learning provides intelligent solutions for identifying attack patterns automatically without relying solely on predefined rules.

The motivation behind this project is to:
- Enhance cybersecurity using AI/ML
- Automate intrusion detection
- Reduce manual monitoring effort
- Improve detection accuracy

---

# 📂 Dataset Description

## Dataset Used
**KDD Cup 1999 Dataset**

## Dataset Features
The dataset contains:
- Network traffic records
- Numerical and categorical features
- Attack labels

### Important Features
- `duration`
- `protocol_type`
- `service`
- `flag`
- `src_bytes`
- `dst_bytes`
- `count`
- `srv_count`

## Attack Classes
- Normal
- DoS (Denial of Service)
- Probe
- R2L (Remote to Local)
- U2R (User to Root)

## Dataset Characteristics
- High-dimensional dataset
- Contains both categorical and numerical features
- Highly imbalanced class distribution

---

# 🔄 Methodology Overview

## 1️⃣ Data Collection
- Loaded training and testing datasets
- Assigned feature column names

## 2️⃣ Exploratory Data Analysis (EDA)
Performed:
- Data inspection
- Statistical analysis
- Class distribution analysis
- Correlation analysis
- Feature distribution visualization

## 3️⃣ Data Preprocessing
- Handled categorical features using One-Hot Encoding
- Scaled numerical features using StandardScaler
- Removed redundant features

## 4️⃣ Feature Selection
Applied:
- Mutual Information Classification
- Correlation-based feature filtering

## 5️⃣ Model Development
Implemented multiple machine learning models:
- Random Forest Classifier
- XGBoost Classifier
- Multi-Layer Perceptron (MLP)

## 6️⃣ Model Evaluation
Evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC Score
- Confusion Matrix

---

# 📊 Results Summary

| Model | Accuracy | Key Observation |
|---|---|---|
| Random Forest | High Accuracy | Strong performance on majority classes |
| XGBoost | Very High Accuracy | Best overall classification performance |
| MLP Classifier | Good Accuracy | Captured nonlinear attack patterns |

## Key Findings
- Feature engineering significantly improved model performance.
- Ensemble models performed better than basic classifiers.
- Service-related features were highly informative.
- Class imbalance affected minority attack detection.

---

# 🖼️ Screenshots / Application Images

Add your application screenshots inside an `images` folder.

Example:

```markdown
![Home Page](images/home.png)

![Prediction Page](images/prediction.png)

![Model Performance](images/results.png)
```

---

# ⚙️ Installation and Setup

## Clone the Repository
```bash
git clone https://github.com/your-username/your-repository-name.git
```

## Navigate to Project Directory
```bash
cd your-repository-name
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run the Application
```bash
streamlit run app.py
```

---

# 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit

---

# 🚀 Streamlit Deployment

Live Application Link:

```markdown
https://intrusion-detection-network.streamlit.app/
```

---

# 📈 Future Improvements
- Use modern cybersecurity datasets
- Apply Deep Learning models
- Real-time intrusion detection
- Hyperparameter optimization
- Cloud deployment

---

# 📚 References
1. KDD Cup 1999 Dataset  
2. Scikit-learn Documentation  
3. XGBoost Documentation  
4. Streamlit Documentation  

---

# ⭐ Conclusion
This project demonstrates how Machine Learning can be effectively applied for network intrusion detection. By combining data preprocessing, feature engineering, and advanced classification models, the system can accurately identify malicious network activities and contribute to improved cybersecurity monitoring.
