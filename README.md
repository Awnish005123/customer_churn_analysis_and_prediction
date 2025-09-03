[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.x-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imblearn-SMOTE-43B02A?style=flat&logo=python&logoColor=white)](https://imbalanced-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EB4335?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-9ACD32?style=flat&logo=lightning&logoColor=white)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-FFCC00?style=flat&logo=cat&logoColor=black)](https://catboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-0C2340?style=flat&logo=python&logoColor=white)](https://shap.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-7B1FA2?style=flat&logo=opsgenie&logoColor=white)](https://optuna.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat&logo=plotly&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Stat%20plots-4C9A2A?style=flat&logo=databricks&logoColor=white)](https://seaborn.pydata.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-Stats-003B57?style=flat&logo=scipy&logoColor=white)](https://www.statsmodels.org/)

## 📂 Overview

Predict customer attrition (churn) for a credit-card portfolio with an end‑to‑end Python workflow: EDA, statistical testing, preprocessing, feature engineering, class balancing, model training, evaluation, ensembling, and SHAP‑based explainability on BankChurners.csv.

- Target: `Attrition_Flag` (imbalanced ~15–16%)
- Key signals: transactions, utilization, inactivity months, contact count, relationship depth, engineered ratios

## Key objectives
- Build robust classifiers to flag high‑risk churners early.
- Quantify behavioral drivers of churn via SHAP and statistical tests.
- Output actionable retention recommendations by segment.

## Data
- Source file: `BankChurners.csv`
- Two non‑informative columns (index 21, 22) dropped
- Mixed numeric/categorical schema; explicit dtype optimization for memory/runtime
- Stratified train/test split using transformed transaction amount bins

**Key Features**

| Feature Name               | Data Type | Category    | Description                                           |
| -------------------------- | --------- | ----------- | ----------------------------------------------------- |
| `CLIENTNUM`                | int64     | Identifier  | Unique customer ID (not used for modeling)            |
| `Attrition_Flag`           | object    | Target      | Churn status: 🟢 Existing or 🔴 Attrited              |
| `Customer_Age`             | int64     | Numerical   | Age of the customer                                   |
| `Gender`                   | object    | Categorical | Customer gender                                       |
| `Dependent_count`          | int64     | Numerical   | Number of dependents                                  |
| `Education_Level`          | object    | Categorical | Education level (High School, Graduate, etc.)         |
| `Marital_Status`           | object    | Categorical | Marital status (Married, Single, etc.)                |
| `Income_Category`          | object    | Categorical | Income bracket (Less than \$40K, \$40K - \$60K, etc.) |
| `Card_Category`            | object    | Categorical | Credit card type (Blue, Silver, Gold, Platinum)       |
| `Months_on_book`           | int64     | Numerical   | Tenure with the bank (in months)                      |
| `Total_Relationship_Count` | int64     | Numerical   | Total number of bank products held                    |
| `Months_Inactive_12_mon`   | int64     | Numerical   | Inactive months in the past 12 months                 |
| `Contacts_Count_12_mon`    | int64     | Numerical   | Customer service contacts in the past 12 months       |
| `Credit_Limit`             | float64   | Numerical   | Credit card limit                                     |
| `Total_Revolving_Bal`      | int64     | Numerical   | Revolving balance on the card                         |
| `Avg_Open_To_Buy`          | float64   | Numerical   | Average available credit                              |
| `Total_Trans_Amt`          | int64     | Numerical   | Total transaction amount in last 12 months            |
| `Total_Trans_Ct`           | int64     | Numerical   | Total transaction count in last 12 months             |
| `Total_Ct_Chng_Q4_Q1`      | float64   | Numerical   | Change in transaction count Q4 vs Q1                  |
| `Total_Amt_Chng_Q4_Q1`     | float64   | Numerical   | Change in transaction amount Q4 vs Q1                 |
| `Avg_Utilization_Ratio`    | float64   | Numerical   | Average card utilization rate                         |



## Methods
- EDA and statistics: distribution plots, correlation heatmaps, outlier checks (IQR), Chi‑Square, Mann–Whitney U, ANOVA/Kruskal, Shapiro/Anderson/Normaltest, VIF
- Preprocessing: `ColumnTransformer` with `SimpleImputer`, `RobustScaler`/`StandardScaler`, `OneHotEncoder`
- Skew handling: Yeo‑Johnson transforms; sparse‑aware bin+log for near‑zero columns
- Imbalance: `SMOTE` applied on training folds for minority upsampling
- Evaluation: Stratified K‑fold CV with ROC‑AUC primary; PR‑AUC, confusion matrix, class metrics supplemental

<details>
  <summary><b>Feature engineering (expand)</b></summary>

- Utilization_Ratio_Per_Trans = Avg_Utilization_Ratio / (Total_Trans_Ct + 1)  
- Credit_Usage_Efficiency = Total_Trans_Amt / (Credit_Limit + 1)  
- Trans_Change_Rate = Total_Ct_Chng_Q4_Q1 / (Total_Trans_Ct + 1)  
- Age_To_Months_Ratio = Customer_Age / (Months_on_book + 1)  
- RFM segmentation features + labeled segment for modeling and BI
</details>

## Models
- Baselines: LogisticRegression, LinearSVC/SVC, KNN, GaussianNB, DecisionTree, RandomForest, ExtraTrees, AdaBoost, GradientBoosting, MLP, Ridge/RidgeCV
- Gradient boosting trio (tuned): CatBoost, LightGBM, XGBoost (hyperparameters via Optuna)
- Ensemble: Soft Voting (CatBoost + LGBM + XGBoost) to maximize ROC‑AUC and stabilize generalization

## Results

**Top 5 Most Influential Features**

| Feature                                         | Interpretation                                                                                                                           |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **num\_robust\_\_Total\_Trans\_Ct**             | Transaction count is the **most important factor**. More frequent transactions (red) lead to higher credit score predictions.            |
| **num\_robust\_\_PT\_Total\_Trans\_Amt**        | Total transaction amount (proportional). Higher spending indicates stronger financial behavior, contributing positively to credit score. |
| **num\_standard\_\_Total\_Revolving\_Bal**      | Total revolving balance. High values (red) lead to negative SHAP values — indicating potential risk due to high debt levels.             |
| **num\_robust\_\_PT\_Total\_Ct\_Chng\_Q4\_Q1**  | Change in transaction count between Q4 and Q1. Sudden increases or drops reflect financial volatility and can influence score both ways. |
| **num\_robust\_\_PT\_Total\_Amt\_Chng\_Q4\_Q1** | Change in transaction amount between quarters. The impact is bidirectional depending on the direction and magnitude of change.           |

**Other Notable Features:**

| Feature                                    | Interpretation                                                                                                              |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `cat_onehot__Months_Inactive_12_mon_1`     | Customers inactive for 1 month in the past year show strong negative impact (red dots shifting left).                       |
| `num_robust__Customer_Age`                 | Age has mild influence. Older customers tend to have positive SHAP values, possibly reflecting more stable credit behavior. |
| `num_standard__PT_Credit_Usage_Efficiency` | Efficiency of credit usage — higher efficiency (red) contributes positively to credit score.                                |
| `cat_keep__Gender`                         | Minimal effect — the model doesn’t significantly differentiate by gender.                                                   |
| `num_standard__PT_Credit_Limit`            | Credit limit has minor influence but follows expected direction: higher limit → higher predicted score.                     |

**General Takeaways:**

* **Credit behavior** (transactions, balances, quarter-over-quarter changes) is the **dominant signal** in determining churn.
* Categorical features like one-hot encoded `inactive months` and `relationship count` also provide meaningful patterns.
* **Demographics** like gender or marital status play a **less important role**, indicating the model focuses more on **behavioral and financial indicators**.

**Summary Table – SHAP Importance & Statistical Significance**

| Feature                        | SHAP Importance | Stat Test Significance | Combined Insight                                                    | Suggested Action                                                    |
| ------------------------------ | --------------- | ---------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Total\_Trans\_Ct**           | Very High    | ✅ Significant          | Fewer transactions → higher churn risk                              | Monitor customers with low activity; send offers to re-engage usage |
| **Total\_Trans\_Amt**          | Very High    | ✅ Significant          | Lower spending → associated with higher churn                       | Encourage spend with personalized promotions                        |
| **Total\_Revolving\_Bal**      | High         | ✅ Significant          | Low balance may reflect card inactivity → churn risk                | Target low-usage customers with usage incentives                    |
| **Total\_Ct\_Chng\_Q4\_Q1**    | Medium       | ✅ Significant          | Smaller increase in transaction count → churn-prone                 | Flag drop in transactional patterns; re-engagement campaigns        |
| **Total\_Amt\_Chng\_Q4\_Q1**   | Medium       | ✅ Significant          | Low spending growth over time → churn signal                        | Promote positive spending trends through milestone offers           |
| **Avg\_Open\_To\_Buy**         | Low          | ✅ Significant          | Less available credit → potential disengagement                     | Consider offering alternative card options                          |
| **Contacts\_Count\_12\_mon**   | Low          | ✅ Significant          | Frequent contact → correlated with churn (possibly dissatisfaction) | Balance communication; avoid overwhelming the customer              |
| **Months\_Inactive\_12\_mon**  | Medium       | ✅ Significant          | 3–4 inactive months → strong churn predictor                        | Launch reactivation campaigns for inactive users                    |
| **Total\_Relationship\_Count** | Medium       | ✅ Significant          | Fewer linked products (1–2) → less loyal customers                  | Cross-sell/upsell to increase engagement and stickiness             |

**Overall Observations:**

* Strong alignment between **SHAP-based feature importance** and **statistical test results** reinforces trust in model behavior.
* Behavioral indicators (transactions, usage changes) dominate both SHAP impact and statistical correlation.
* Demographic features like `Gender`, `Marital_Status`, and `Education_Level` show low predictive value, validating the model’s behavioral focus.

**Strategic Recommendations:**

| Behavioral Segment              | Key Features                        | Recommended Strategy                                         |
| ------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| **Low activity users**          | `Total_Trans_Ct`, `Total_Trans_Amt` | Use cashback or bonus incentives tied to usage thresholds    |
| **Declining engagement**        | `*_Chng_Q4_Q1`                      | Early alerts + customer support follow-up                    |
| **Limited product holders**     | `Total_Relationship_Count`          | Introduce new product bundles or upgrade options             |
| **Inactive users (3–4 months)** | `Months_Inactive_12_mon`            | Trigger personalized reactivation flows                      |
| **Over-contacted customers**    | `Contacts_Count_12_mon`             | Optimize communication frequency and tailor outreach content |

## Clone and run
git clone https://github.com/Awnish005123/customer_churn_analysis_and_prediction.git

MIT License | ⭐ Star if helpful! 🏥✨
