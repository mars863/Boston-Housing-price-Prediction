
# Boston Housing Price Prediction

## 📌 Project Overview
This project predicts house prices using the Boston Housing dataset. Multiple regression algorithms were trained and compared to identify the best-performing model. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, hyperparameter tuning, and model serialization.

---

## 🎯 Objectives

- Predict house prices using machine learning.
- Compare multiple regression algorithms.
- Select the best-performing model.
- Improve performance through hyperparameter tuning.
- Save the trained model for future predictions.

---

## 📂 Dataset

The dataset contains housing-related features including:

- CRIM
- ZN
- INDUS
- CHAS
- NOX
- RM
- AGE
- DIS
- RAD
- TAX
- PTRATIO
- B
- LSTAT

**Target Variable:** MEDV (Median value of owner-occupied homes)

---

## 🛠 Technologies Used

- Python
- Jupyter Notebook / Google Colab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

---

## 🔄 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Missing Value Handling
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Hyperparameter Tuning
10. Model Serialization

---

## 🤖 Models Compared

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regressor (SVR)
- XGBoost Regressor

**Best Model:** XGBoost Regressor

---

## 📊 Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## ⚙️ Hyperparameter Tuning

RandomizedSearchCV was used to optimize:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- gamma

---

## 📈 Results

- Compared six regression models.
- XGBoost achieved the best overall performance.
- Improved the model through hyperparameter tuning.
- Saved the trained model using Joblib.

---

## 🚀 Future Improvements

- Deploy using Streamlit or FastAPI.
- Use larger datasets.
- Experiment with advanced feature engineering.

---

## 👨‍💻 Author

**Arshad Alom**

Machine Learning Internship Project
