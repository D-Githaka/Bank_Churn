# Bank Customer Churn Prediction: Exploratory Analysis & Prediction

This project aims to tackle the problem of customer churn in a bank by first conducting a detailed exploratory data analysis to understand the factors influencing churn and then building a predictive model to identify customers likely to churn. The goal is to provide insights and a tool that can help the bank proactively retain valuable customers.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Project Structure](#project-structure)
3.  [Dataset](#dataset)
4.  [Exploratory Data Analysis (EDA) Highlights](#exploratory-data-analysis-eda-highlights)
    *   [Churn Proportion](#churn-proportion)
    *   [Categorical Feature Relationships](#categorical-feature-relationships)
    *   [Continuous Feature Relationships](#continuous-feature-relationships)
5.  [Feature Engineering](#feature-engineering)
6.  [Data Preprocessing](#data-preprocessing)
7.  [Modeling and Evaluation](#modeling-and-evaluation)
    *   [Models Explored](#models-explored)
    *   [Evaluation Metric](#evaluation-metric)
    *   [Best Model](#best-model)
8.  [Streamlit Web Application](#streamlit-web-application)
9.  [Setup](#setup)
10. [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
11. [Project Development (Using the Notebook)](#project-development-using-the-notebook)
12. [Future Work](#future-work)
13. [Contributing](#contributing)
14. [License](#license)
15. [Acknowledgements](#acknowledgements)

## 1. Project Overview

Customer churn is a critical challenge for banks. Losing existing customers can be more costly than acquiring new ones. This project addresses this by:
*   Analyzing customer data to uncover patterns and factors associated with churn.
*   Developing a machine learning model to predict customer churn risk.
*   Deploying a user-friendly web application to facilitate predictions and showcase insights.

The primary objective is to identify customers at high risk of churning so that targeted retention strategies can be implemented.

## 2. Project Structure

The repository contains the following key files and directories:

- app.py
- Bank_Churn_Prediction.ipynb
- best_model.pkl
- Churn_Modeling.csv
- LICENSE
- min_max_values.pkl
- README.html
- README.md
- requirement.txt
- train.csv


## 3. Dataset

The project utilizes the "Predicting Churn for Bank Customers" dataset. It contains information about 10,000 bank customers, including demographic data, account details, and whether they have exited (churned) the bank.

**Attributes:**

*   `RowNumber`, `CustomerId`, `Surname`: Identifiers (dropped during preprocessing).
*   `CreditScore`: Customer's credit score.
*   `Geography`: Customer's country of residence (France, Germany, Spain).
*   `Gender`: Customer's gender.
*   `Age`: Customer's age.
*   `Tenure`: Number of years the customer has been with the bank.
*   `Balance`: Account balance.
*   `NumOfProducts`: Number of bank products (e.g., checking account, savings account) the customer has.
*   `HasCrCard`: Whether the customer has a credit card (1=Yes, 0=No).
*   `IsActiveMember`: Whether the customer is an active member (1=Yes, 0=No).
*   `EstimatedSalary`: Estimated salary of the customer.
*   `Exited`: Target variable - whether the customer churned (1=Yes, 0=No).

The dataset was found to have **no missing values** initially, which is a rare and positive finding.

## 4. Exploratory Data Analysis (EDA) Highlights

The EDA phase aimed to understand the relationships between the features and the target variable (`Exited`). Key findings include:

### Churn Proportion

*   Approximately **20%** of the customers in the dataset have churned, indicating an imbalanced dataset. The modeling approach needs to consider this imbalance, focusing on the 'Churned' class (1).

### Categorical Feature Relationships

*   **Geography:** While France has the most customers, customers from Germany and Spain show a higher proportion of churn relative to their population size. This suggests potential regional issues or differences in customer base/service.
*   **Gender:** A higher proportion of female customers churn compared to male customers.
*   **HasCrCard:** Interestingly, a slight majority of customers who churned possessed a credit card. Given most customers have credit cards, this might be less impactful than other factors, but warrants further investigation.
*   **IsActiveMember:** As expected, inactive members have a significantly higher churn rate than active members. The high overall number of inactive members is a concern for the bank.

### Continuous Feature Relationships

*   **CreditScore:** No significant difference was observed in credit score distribution between churned and retained customers.
*   **Age:** Older customers show a higher likelihood of churning compared to younger ones, suggesting age-specific service preferences or needs.
*   **Tenure:** Both very new customers and long-term customers exhibit slightly higher churn rates compared to those with average tenure.
*   **Balance:** Worryingly, customers with higher bank balances are more likely to churn. This impacts the bank's capital significantly.
*   **NumOfProducts:** Customers with 3 or 4 products show a very high churn rate, likely indicating dissatisfaction or complexity. Customers with 1 or 2 products have lower churn rates.
*   **EstimatedSalary:** No significant relationship was found between estimated salary and churn likelihood.

## 5. Feature Engineering

Based on the EDA, new features were engineered to potentially capture more complex relationships:

*   **`BalanceSalaryRatio`**: Ratio of account balance to estimated salary. This feature showed that customers with a higher balance-to-salary ratio tend to churn more, despite estimated salary alone not being a strong predictor.
*   **`TenureByAge`**: Ratio of tenure to customer age (adjusted by subtracting 18). This feature aimed to standardize tenure relative to an individual's adult life span.
*   **`CreditScoreGivenAge`**: Ratio of credit score to customer age (adjusted by subtracting 18). This aimed to capture credit behavior relative to age.

## 6. Data Preprocessing

The data underwent the following preprocessing steps before modeling:

*   **Dropping Irrelevant Columns:** `RowNumber`, `CustomerId`, and `Surname` were removed.
*   **Train-Test Split:** The dataset was split into 80% for training and 20% for testing.
*   **Handling Binary Categorical Variables:** `HasCrCard` and `IsActiveMember` were transformed from {0, 1} to {-1, 1} to allow models (like linear models) to potentially capture negative relationships.
*   **One-Hot Encoding:** Categorical variables (`Geography`, `Gender`) were one-hot encoded, mapping each category to a new binary feature ({1, -1}).
*   **Feature Scaling:** Continuous features (including the engineered ones) were scaled using Min-Max scaling to a range between 0 and 1. This is important for distance-based models like SVM and helps improve convergence for other models.
*   **Handling `NaN` / `Inf`:** Steps were taken to identify and handle potential `NaN` or `Inf` values introduced during feature engineering (especially from division) by replacing them with 0.

A `DfPrepPipeline` function was developed to ensure these steps can be consistently applied to both the training data and new data for prediction.

## 7. Modeling and Evaluation

Several machine learning classification models were trained and evaluated on the preprocessed training data.

### Models Explored

*   Logistic Regression (Primal space and with Polynomial Kernel)
*   Support Vector Machines (SVM) (RBF and Polynomial Kernels)
*   Random Forest Classifier
*   XGBoost Classifier

Hyperparameter tuning was performed using `GridSearchCV` with cross-validation for each model.

### Evaluation Metric

Given the imbalance in the dataset and the business objective of identifying customers likely to churn, the primary evaluation focus was on the metrics for the positive class (churn = 1), specifically:

*   **Recall (Sensitivity):** The ability of the model to find all the relevant cases (correctly identify all churned customers). This is crucial for not missing potential churners.
*   **Precision:** The ability of the model to return only relevant cases (the proportion of identified churners who actually churned). This is important for efficiently allocating retention resources.
*   **F1-Score:** The harmonic mean of Precision and Recall, providing a balance between the two.
*   **ROC AUC Score:** Measures the model's ability to distinguish between the positive and negative classes.

### Best Model

Based on the balance between Recall and Precision for the 'Churned' class on the training data, the **Random Forest Classifier** emerged as the most suitable model for this problem.

*   **Training Performance (Random Forest):**
    *   Precision (Churn=1): ~0.88
    *   Recall (Churn=1): ~0.53
    *   F1-Score (Churn=1): ~0.66
    *   Overall Accuracy: High (~0.95)
    *   ROC AUC: High

While the recall indicates that the model identifies about half of the actual churners, the high precision means that when the model predicts a customer will churn, it's correct about 88% of the time. This is a valuable starting point for targeted retention efforts.

The model and the preprocessing parameters (`best_model.pkl`, `train_cols.pkl`, `min_max_values.pkl`) were saved after training.

## 8. Streamlit Web Application

The project includes an interactive Streamlit app (`app.py`) that demonstrates the model's capabilities.

The app allows users to:

*   Input individual customer details via a form and get an instant churn prediction and probability.
*   Upload a CSV file with multiple customer records for batch prediction. The results, including predicted churn status and probability, are displayed and can be downloaded.
*   Visualize the feature importances learned by the Random Forest model.
*   View key EDA plots derived from the original dataset to understand general customer characteristics related to churn.

## 9. Setup

To set up the project locally:

1.  **Clone the Repository:**
*(Replace `D-Githaka and `Churn_Prediction`.)*

2.  **Install Dependencies:**
    Using a virtual environment is highly recommended.

    *   **Using `venv`:**
        # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    *   **Using `conda`:**
    Install packages:
    3.  **Obtain Data and Saved Resources:**
    *   Download the `Churn_Modelling.csv` dataset.
    *   Place `Churn_Modelling.csv` in the root directory of the project.
    *   Ensure `best_model.pkl`, `train_cols.pkl`, and `min_max_values.pkl` are present in the project root. If you cloned the repository and these were committed, they should be there. If not, you must run the notebook first to generate them.

## 10. How to Run the Streamlit App

1.  Ensure you have completed the [Setup](#9-setup) steps and activated your virtual environment.
2.  Open your terminal and navigate to the project root directory (`your-churn-prediction-project/`).
3.  Run the Streamlit app:
4.  Your web browser should open the app (typically at `http://localhost:8501`).

## 11. Project Development (Using the Notebook)

The `your_notebook.ipynb` file is the core of the analysis and model training. You can open and run this notebook using Jupyter Notebook or JupyterLab (installed via `pip install notebook` or `pip install jupyterlab`). It details the entire process from data loading to saving the final model and preprocessing objects.

## 12. Future Work

*   Explore more advanced feature engineering techniques.
*   Experiment with different modeling algorithms (e.g., LightGBM, CatBoost, Neural Networks).
*   Address dataset imbalance using techniques like SMOTE or different class weighting in the model.
*   Tune model hyperparameters more extensively.
*   Integrate SHAP or LIME for model interpretability to explain individual predictions.
*   Containerize the application using Docker for easier deployment.
*   Set up CI/CD pipelines for automated testing and deployment.

## 13. Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## 14. License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## 15. Acknowledgements

*   Dataset Source: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers
*   Libraries Used: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, joblib, streamlit
*   Created by: Denis Githaka