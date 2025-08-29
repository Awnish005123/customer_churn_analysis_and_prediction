# 📂 Overview

**Background**

This dataset contains **credit card customer data** from Kaggle. It includes real-world customer behavior with features related to demographics, credit usage, and account activity.

This binary classification problem aims to **predict whether a customer will churn** based on their profile and activity.

**Goal of the Project**

Build a machine learning model to **predict whether a customer will leave the credit card service** (`Attrition_Flag`: Attrited/Existing).

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



**Project Objective**

The goal of this notebook is to **analyze customer behavior and predict churn**, supporting business decisions like:

* Targeted retention strategies
* Personalized offers for at-risk customers
* Reducing customer attrition

**Key Steps**

* **Exploratory Data Analysis (EDA):** <br>
  Understand patterns in customer behavior and churn.

* **🛠 Feature Engineering:**
  Encode categorical variables, scale numerical features, and create meaningful derived variables (e.g. utilization ratios, transaction trends).

* **Modeling:**
  Apply various classifiers like:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * LightGBM
  * MLPClassifier

* **Evaluation Framework:**

  * Use **Stratified Cross-Validation**
  * Assess using:

    * Accuracy
    * Precision
    * Recall
    * F1-score
    * ROC-AUC