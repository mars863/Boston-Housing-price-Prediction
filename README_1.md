# Details
intern ID =	CITS4273
Full Name =	Arshad Alom
No. of Weeks =	6 Weeks
Project Name = Boston Housing Price Prediction Using XGBoost

Project Scope
Developed an end-to-end machine learning regression model to predict Boston housing prices using the XGBoost Regressor. The project involved data cleaning by handling missing values using median and mode imputation, exploratory data analysis (EDA), correlation analysis, outlier detection using the IQR method, and feature engineering through feature creation, log transformation, and one-hot encoding. Multiple regression algorithms including Linear Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Regression (SVR), and XGBoost were trained and compared. The XGBoost model was further optimized using RandomizedSearchCV with 5-fold cross-validation. Model performance was evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score. Feature importance analysis, actual vs. predicted visualization, and residual analysis were performed to interpret model performance. Finally, the optimized model was serialized using Joblib for future deployment and inference.

# 🏠 Boston Housing Price Prediction

Predicting median house values in the Boston area using classic regression algorithms, with XGBoost — tuned via `RandomizedSearchCV` — as the final model.

---

## 📌 Overview

This project walks through a full regression workflow on the Boston Housing dataset: cleaning missing values, exploratory data analysis, feature engineering, training and comparing six regression models, hyperparameter tuning the winner, and serializing it for reuse.

| | |
|---|---|
| **Task** | Regression |
| **Target** | `MEDV` — median value of owner-occupied homes ($1000s) |
| **Rows / Features** | 506 rows × 13 input features |
| **Best model** | XGBoost Regressor (tuned) |
| **Best R² (test set)** | **0.905** |

---

## 📂 Repository Contents

| File | Description |
|---|---|
| `Boston.ipynb` | Full notebook: EDA → feature engineering → model training → tuning → evaluation |
| `HousingData.csv` | Raw dataset (506 rows, 13 features + target, contains missing values) |
| `house_price_model.joblib` | Final tuned XGBoost model, serialized with `joblib` |
| `Boston_Housing_README.md` | Original project brief |

---

## 📊 Dataset

The dataset contains 506 records describing housing in the Boston area, with the following features:

| Feature | Description |
|---|---|
| `CRIM` | Per-capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres per town |
| `CHAS` | Charles River dummy variable (1 if tract bounds the river) |
| `NOX` | Nitric oxide concentration (parts per 10 million) |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to five Boston employment centers |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Full-value property tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | 1000(Bk − 0.63)² where Bk is the proportion of Black residents by town |
| `LSTAT` | % lower-status population |
| `MEDV` | **Target** — median home value ($1000s) |

`CRIM`, `ZN`, `INDUS`, `AGE`, and `LSTAT` each contained 20 missing values, handled during preprocessing (see below).

> **Note:** This dataset uses the classic `sklearn`-style Boston Housing schema, including the `B` feature. It's used here purely for regression-technique practice; the `B` feature reflects a now-deprecated/controversial construction from the original 1978 source and is not recommended for real-world deployment. See scikit-learn's [note on the dataset's removal](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) for context.

---

## 🔄 Workflow

1. **Data Cleaning** — filled numeric column gaps (`CRIM`, `ZN`, `INDUS`, `AGE`, `LSTAT`) with the median, and `CHAS` with the mode.
2. **EDA** — distributions, correlation heatmaps, boxplots, and pairplots to understand feature relationships with `MEDV`.
3. **Outlier Detection** — flagged outliers per feature using the IQR method.
4. **Feature Engineering**
   - `RM_PER_TAX` = `RM / (TAX + 1)`
   - `CRIM_PER_ROOM` = `CRIM / (RM + 1)`
   - `RM_NOX` = `RM * NOX`
   - `DIS_AGE` = `DIS * AGE`
   - `AGE_CATEGORY` — binned `AGE` into `New` / `Medium` / `Old`, one-hot encoded
   - Log-transformed `CRIM` and `LSTAT` (`log1p`) to reduce skew
5. **Train/Test Split** — 80/20, `random_state=42`
6. **Model Training** — six regressors trained and compared
7. **Hyperparameter Tuning** — `RandomizedSearchCV` (20 iterations, 5-fold CV) on XGBoost
8. **Evaluation** — MAE, MSE, RMSE, R², plus actual-vs-predicted and residual plots
9. **Serialization** — final model saved with `joblib`

---

## 🤖 Model Comparison (Test Set)

| Model | MSE | R² Score |
|---|---|---|
| **XGBoost** | **7.55** | **0.897** |
| Random Forest | 7.89 | 0.892 |
| Decision Tree | 12.88 | 0.824 |
| Linear Regression | 18.49 | 0.748 |
| KNN | 23.69 | 0.677 |
| SVR | 52.10 | 0.290 |

XGBoost and Random Forest clearly outperformed the linear and distance-based models, so XGBoost was selected for tuning.

---

## ⚙️ Hyperparameter Tuning

`RandomizedSearchCV` (20 iterations, 5-fold CV, scoring = R²) searched over:

```python
param_dist = {
    'n_estimators':     [100, 200, 300, 500],
    'max_depth':         [3, 4, 5, 6, 8],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'subsample':         [0.8, 0.9, 1.0],
    'colsample_bytree':  [0.8, 0.9, 1.0],
    'gamma':             [0, 0.1, 0.3, 0.5]
}
```

**Best parameters found:**

```python
{'subsample': 0.8, 'n_estimators': 200, 'max_depth': 4,
 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 0.9}
```

Best cross-validated R²: **0.841**

---

## 📈 Final Model Performance (Test Set)

| Metric | Value |
|---|---|
| MAE | 1.93 |
| MSE | 6.97 |
| RMSE | 2.64 |
| **R²** | **0.905** |

Tuning improved test R² from 0.897 (default XGBoost) to 0.905, and reduced MSE from 7.55 to 6.97.

Diagnostic plots included in the notebook:
- **Feature importance** (XGBoost gain-based importance)
- **Actual vs. Predicted** scatter plot
- **Residual plot** to check for systematic bias

---

## 🚀 Getting Started

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib
```

### Run the notebook

```bash
jupyter notebook Boston.ipynb
```

### Load the trained model for inference

```python
import joblib
import pandas as pd

model = joblib.load("house_price_model.joblib")

# X_new must have the same engineered features/columns used at train time:
# CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT,
# RM_PER_TAX, CRIM_PER_ROOM, NOX (log-transformed CRIM/LSTAT),
# RM_NOX, DIS_AGE, AGE_CATEGORY_Medium, AGE_CATEGORY_Old
predictions = model.predict(X_new)
```

> ⚠️ The model expects the **engineered feature set**, not the raw CSV columns — replicate the preprocessing/feature-engineering steps from the notebook (Section: *Feature Engineering*) before calling `.predict()`.

---

## 🛠 Tech Stack

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Joblib

---

## 🚧 Future Improvements

- Deploy the model behind a REST API (FastAPI) or a Streamlit demo app
- Wrap preprocessing + feature engineering in a `Pipeline`/`ColumnTransformer` so raw input can be fed directly to the model
- Validate on a larger, more current real-estate dataset
- Add SHAP-based explainability for individual predictions

---

## 👨‍💻 Author

**Arshad Alom**
Machine Learning Internship Project

---

## 📄 License

Add a license of your choice (e.g., MIT) if you plan to share or reuse this project publicly.
