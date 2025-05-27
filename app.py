# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:00:45 2025

@author: Denis Githaka
"""

# REQUIRED LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
import joblib # To save and load the model
import matplotlib.pyplot as plt
import seaborn as sns
# Assuming you might use PolynomialFeatures, LogisticRegression, SVC, XGBClassifier if you add options later
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Ensure XGBoost is imported if using it
import warnings
warnings.filterwarnings('ignore')

# Suppress specific warnings if necessary
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration and Load Resources ---

st.set_page_config(layout="wide") # Use wide layout

st.title('Bank Customer Churn Prediction')
st.write('Predicting whether bank customers will churn based on their attributes.')

# Load the trained model and preprocessing data
@st.cache_resource
def load_resources(model_path, cols_path, minmax_path):
    """Loads the trained model, training column names, and MinMax scaler values."""
    try:
        model = joblib.load(model_path)
        train_cols = joblib.load(cols_path)
        min_max_values = joblib.load(minmax_path)
        minVec, maxVec = min_max_values
        st.success("Model and preprocessing data loaded successfully.")
        return model, train_cols, minVec, maxVec
    except FileNotFoundError as e:
        st.error(f"Error: Resource file not found: {e}. Please ensure 'best_model.pkl', 'train_cols.pkl', and 'min_max_values.pkl' are in the same directory.")
        st.stop() # Stop the app if resources are not found
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

# Define paths - Make sure these match where you saved the files
model_path = 'best_model.pkl'
train_cols_path = 'train_cols.pkl'
min_max_path = 'min_max_values.pkl'

# Load resources at the start
model, train_cols, minVec, maxVec = load_resources(model_path, train_cols_path, min_max_path)

# --- Data Preprocessing Pipeline ---

# Redefine the DfPrepPipeline function to be robust for single row or multiple rows
def DfPrepPipeline(df_predict, df_train_Cols, minVec, maxVec, is_training_data=False):
    """
    Preprocesses the input DataFrame for prediction, applying the same steps
    used for training data. Handles both single-row and multi-row DataFrames.
    """

    # Add new engineered features
    # Handle potential division by zero that might result in inf or NaN
    # Use .loc to avoid SettingWithCopyWarning
    df_predict.loc[:, 'BalanceSalaryRatio'] = np.where(df_predict['EstimatedSalary'] == 0, 0, df_predict['Balance'] / df_predict['EstimatedSalary'])
    # Handle potential division by zero or negative age-18
    df_predict.loc[:, 'TenureByAge'] = np.where((df_predict['Age'] - 18) <= 0, 0, df_predict['Tenure'] / (df_predict['Age'] - 18))
    df_predict.loc[:, 'CreditScoreGivenAge'] = np.where((df_predict['Age'] - 18) <= 0, 0, df_predict['CreditScore'] / (df_predict['Age'] - 18))


    # Define variables (ensure consistency with training)
    continuous_vars = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge'
    ]
    cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender']

    # Create a dummy DataFrame to ensure all expected columns are present in the output
    # This is crucial for models expecting a fixed number of features
    # Include 'Exited' only if processing potential training data or if it's in the original columns
    expected_initial_cols = [col for col in df_train_Cols if col in (['Exited'] + continuous_vars + cat_vars)]
    # Ensure all initial columns from the training data are present in the prediction data
    for col in expected_initial_cols:
        if col not in df_predict.columns:
            df_predict[col] = None # Add missing column with None/NaN initially


    # Select relevant columns from the input DataFrame
    df_predict_processed = df_predict[expected_initial_cols].copy()


    # Replace 0 with -1 for binary categorical variables
    # Use .loc to avoid SettingWithCopyWarning and ensure existence
    if 'HasCrCard' in df_predict_processed.columns:
        df_predict_processed.loc[:, 'HasCrCard'] = df_predict_processed['HasCrCard'].replace(0, -1)
    if 'IsActiveMember' in df_predict_processed.columns:
        df_predict_processed.loc[:, 'IsActiveMember'] = df_predict_processed['IsActiveMember'].replace(0, -1)

    # One-hot encode categorical columns with 1 and -1
    lst = ['Geography', 'Gender']
    for col in lst:
        if col in df_predict_processed.columns:
            # Get unique values from the TRAINING data for this column to ensure all possible OHE columns are created
            # We need to load the original training data or save its unique categorical values
            # For simplicity here, let's assume the training data had 'France', 'Germany', 'Spain' for Geography
            # and 'Female', 'Male' for Gender. In a real application, you'd load these from saved data.
            if col == 'Geography':
                 training_unique_vals = ['France', 'Germany', 'Spain']
            elif col == 'Gender':
                 training_unique_vals = ['Female', 'Male']
            else:
                 training_unique_vals = df_predict_processed[col].dropna().unique() # Fallback, less robust

            for val in training_unique_vals:
                new_col = f"{col}_{val}"
                # Create the new column and initialize it
                df_predict_processed.loc[:, new_col] = -1 # Initialize with -1

                # Set to 1 if the original value matches
                # Ensure the original column has the correct data type before comparison
                if df_predict_processed[col].dtype.kind in {'O', 'U', 'S'}:
                     df_predict_processed.loc[df_predict_processed[col] == val, new_col] = 1

            df_predict_processed = df_predict_processed.drop(columns=[col]) # Drop the original column

    # Add any missing columns from training set with value -1
    # This ensures that the processed DataFrame has the exact same columns as the training data (features + potentially Exited)
    for col in df_train_Cols:
        if col not in df_predict_processed.columns:
            df_predict_processed[col] = -1


    # Ensure all columns from training set are present and ordered
    df_predict_processed = df_predict_processed[df_train_Cols]

    # Apply Min-Max scaling to continuous variables
    # Ensure continuous_vars list is accurate and columns exist
    current_continuous_vars = [var for var in continuous_vars if var in df_predict_processed.columns]

    # First, handle potential inf/NaN values from division before scaling
    df_predict_processed[current_continuous_vars] = df_predict_processed[current_continuous_vars].mask(np.isinf(df_predict_processed[current_continuous_vars]))
    df_predict_processed[current_continuous_vars] = df_predict_processed[current_continuous_vars].fillna(0) # Fill resulting NaNs with 0

    # Apply scaling - handle potential division by zero in scaling if max == min
    for var in current_continuous_vars:
        if var in minVec and var in maxVec: # Check if scaling values exist for this variable
            min_val = minVec[var]
            max_val = maxVec[var]
            if max_val != min_val:
                # Ensure the column is numeric before scaling
                df_predict_processed.loc[:, var] = pd.to_numeric(df_predict_processed[var], errors='coerce').fillna(0)
                df_predict_processed.loc[:, var] = (df_predict_processed[var] - min_val) / (max_val - min_val)
            else:
                df_predict_processed.loc[:, var] = 0 # Or handle as appropriate if min==max
        else:
             # If scaling values are missing for a variable, handle as appropriate (e.g., keep original or set to 0)
             # For this app, we expect all continuous_vars from training to have min/max
             df_predict_processed.loc[:, var] = pd.to_numeric(df_predict_processed[var], errors='coerce').fillna(0)


    # Final check for any remaining NaNs after scaling
    df_predict_processed = df_predict_processed.fillna(0)

    # If this is not training data processing, drop the 'Exited' column if it exists
    if not is_training_data and 'Exited' in df_predict_processed.columns:
        df_predict_processed = df_predict_processed.drop(columns=['Exited'])

    # Ensure the columns match exactly the feature columns from training
    feature_cols_from_training = [col for col in df_train_Cols if col != 'Exited']
    # Reindex to ensure column order matches training data features
    df_predict_processed = df_predict_processed.reindex(columns=feature_cols_from_training, fill_value=-1) # Use fill_value for missing new OHE cols


    return df_predict_processed

# --- Feature Importance Visualization ---
# Assuming the best model is a RandomForestClassifier
if isinstance(model, RandomForestClassifier):
    st.sidebar.header("Feature Importance (from Trained Model)")
    feature_importances = pd.Series(model.feature_importances_, index=[col for col in train_cols if col != 'Exited'])
    feature_importances = feature_importances.sort_values(ascending=False)

    fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
    feature_importances.plot(kind='barh', ax=ax_importance)
    ax_importance.set_title('Feature Importance')
    ax_importance.set_xlabel('Importance Score')
    st.sidebar.pyplot(fig_importance)
else:
    st.sidebar.info("Feature importance visualization is only available for Random Forest models.")


# --- Sidebar for Input Type Selection ---
st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select an input method:", ('Single Customer Input', 'Upload CSV for Batch Prediction'))

# --- Main Area Content ---

# --- Single Customer Input ---
if input_method == 'Single Customer Input':
    st.header('Single Customer Churn Prediction')
    st.write("Enter the details of a single customer to predict their churn status.")

    # Input fields for customer attributes
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=650)
        geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
        gender = st.selectbox('Gender', ['Female', 'Male'])
        age = st.number_input('Age', min_value=18, max_value=100, value=40)
        tenure = st.number_input('Tenure (Years with bank)', min_value=0, max_value=10, value=5)

    with col2:
        balance = st.number_input('Account Balance', min_value=0.0, value=50000.0)
        num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
        has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No'])
        is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])
        estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=60000.0)

    # Convert categorical inputs to the format used in training (-1 for No, 1 for Yes)
    has_cr_card_val = 1 if has_cr_card == 'Yes' else -1
    is_active_member_val = 1 if is_active_member == 'Yes' else -1

    # Create a DataFrame from user inputs
    user_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card_val,
        'IsActiveMember': is_active_member_val,
        'EstimatedSalary': estimated_salary
    }

    user_df = pd.DataFrame([user_data])

    # --- Prediction Button for Single Input ---
    if st.button('Predict Churn'):
        # Apply the preprocessing pipeline to the user data
        try:
            processed_user_df = DfPrepPipeline(user_df.copy(), train_cols, minVec, maxVec, is_training_data=False)

            # Make prediction
            prediction = model.predict(processed_user_df)
            prediction_proba = model.predict_proba(processed_user_df)[:, 1] # Probability of churning (class 1)

            # Display result
            st.subheader('Prediction Result')
            if prediction[0] == 1:
                st.error(f"This customer is likely to churn. Probability: {prediction_proba[0]:.2f}")
            else:
                st.success(f"This customer is likely to be retained. Probability: {prediction_proba[0]:.2f}")

            # Optional: Display the processed input data for debugging
            # st.subheader("Processed Input Data (for debugging)")
            # st.write(processed_user_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


# --- Upload CSV for Batch Prediction ---
elif input_method == 'Upload CSV for Batch Prediction':
    st.header('Batch Churn Prediction from CSV')
    st.write("Upload a CSV file containing customer data to get churn predictions.")
    st.info("Please ensure your CSV file has columns with names matching the input fields: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard (0/1), IsActiveMember (0/1), EstimatedSalary.")
    st.warning("The 'HasCrCard' and 'IsActiveMember' columns should contain 0 or 1.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Original Data Preview:")
            st.dataframe(batch_df.head())

            # Validate expected columns are present
            expected_input_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                                   'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                                   'EstimatedSalary']
            if not all(col in batch_df.columns for col in expected_input_cols):
                missing = [col for col in expected_input_cols if col not in batch_df.columns]
                st.error(f"Missing required columns in CSV: {', '.join(missing)}")
                st.stop() # Stop if required columns are missing

            st.write("Processing data for prediction...")

            # Apply the preprocessing pipeline to the batch data
            # Need to handle potential NaNs in the input CSV gracefully
            batch_df_cleaned = batch_df.copy()
            # Fill potential NaNs in numerical columns with 0 or median/mean if appropriate
            numerical_cols_to_fill = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
            for col in numerical_cols_to_fill:
                 if col in batch_df_cleaned.columns:
                    batch_df_cleaned[col] = pd.to_numeric(batch_df_cleaned[col], errors='coerce').fillna(0) # Coerce non-numeric to NaN, then fill NaN

            # Fill potential NaNs in categorical columns with a placeholder or the mode
            categorical_cols_to_fill = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
            for col in categorical_cols_to_fill:
                 if col in batch_df_cleaned.columns:
                     # Fill with a placeholder or mode, depending on desired behavior
                     # Using a placeholder like 'Unknown' or mode can be options
                     # If using mode, calculate it after converting to string to handle potential NaNs
                     if not batch_df_cleaned[col].isnull().all(): # Avoid calculating mode on all NaNs
                         mode_val = batch_df_cleaned[col].mode().iloc[0] if not batch_df_cleaned[col].mode().empty else 'Unknown_Placeholder'
                         batch_df_cleaned[col] = batch_df_cleaned[col].fillna(mode_val)
                     else:
                         batch_df_cleaned[col] = batch_df_cleaned[col].fillna('Unknown_Placeholder') # Fill entirely NaN column


            # Ensure binary categorical columns are 0 or 1 before pipeline
            if 'HasCrCard' in batch_df_cleaned.columns:
                 batch_df_cleaned['HasCrCard'] = batch_df_cleaned['HasCrCard'].astype(int) # Convert to int
            if 'IsActiveMember' in batch_df_cleaned.columns:
                 batch_df_cleaned['IsActiveMember'] = batch_df_cleaned['IsActiveMember'].astype(int) # Convert to int


            # Call the pipeline
            processed_batch_df = DfPrepPipeline(batch_df_cleaned.copy(), train_cols, minVec, maxVec, is_training_data=False)


            st.write("Making predictions...")
            # Make predictions and get probabilities
            predictions = model.predict(processed_batch_df)
            prediction_proba = model.predict_proba(processed_batch_df)[:, 1] # Probability of churning (class 1)

            # Add predictions and probabilities back to the original DataFrame
            batch_df['Churn_Predicted'] = predictions
            batch_df['Churn_Probability'] = prediction_proba

            st.write("Predictions complete!")

            # Display results
            st.subheader("Prediction Results")
            st.dataframe(batch_df)

            # Provide a download link for the results
            csv_output = batch_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction Results as CSV",
                data=csv_output,
                file_name='churn_predictions.csv',
                mime='text/csv',
            )

            # Optional: Display some stats on the predictions
            st.subheader("Prediction Summary")
            churn_counts = batch_df['Churn_Predicted'].value_counts().rename({0: 'Retained', 1: 'Churned'})
            st.write("Count of predicted churn/retained customers:")
            st.bar_chart(churn_counts)


        except Exception as e:
            st.error(f"An error occurred during CSV processing or prediction: {e}")
            st.write("Please check the format and content of your CSV file.")


# --- Exploratory Data Analysis (Optional Section) ---
# You can add a section to display some general EDA plots
# You might want to load a sample of the training data for this, or analyze the uploaded data

st.sidebar.header("Exploratory Data Analysis")
st.sidebar.info("The following plots are based on the characteristics of the original training dataset.")

# Load a small sample of the original data or the full training data if manageable
# Assuming you have saved the original training data (before feature engineering) as 'original_train_data.csv'
try:
    @st.cache_data # Use cache_data for data loading
    def load_original_data(path):
         df = pd.read_csv(path)
         # Apply initial cleaning steps like dropping columns if they were part of the original notebook
         df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1, errors='ignore') # Use errors='ignore' if cols might not exist
         return df

    original_data_path = '/content/drive/My Drive/Churn_Modelling.csv' # Adjust this path if you saved it elsewhere
    original_df = load_original_data(original_data_path)

    st.subheader("Explore Customer Data Characteristics")

    # Example 1: Churn Distribution
    labels = 'Exited', 'Retained'
    sizes = [original_df['Exited'][original_df['Exited']==1].count(), original_df['Exited'][original_df['Exited']==0].count()]
    explode = (0, 0.1)
    fig_churn_dist, ax_churn_dist = plt.subplots(figsize=(6, 6))
    ax_churn_dist.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax_churn_dist.axis('equal')
    ax_churn_dist.set_title("Proportion of customer churned and retained", size = 14)
    st.pyplot(fig_churn_dist)


    # Example 2: Churn vs Geography
    fig_geo, ax_geo = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Geography', hue = 'Exited',data = original_df, ax=ax_geo)
    ax_geo.set_title('Churn by Geography')
    st.pyplot(fig_geo)

    # Example 3: Churn vs Gender
    fig_gender, ax_gender = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Gender', hue = 'Exited',data = original_df, ax=ax_gender)
    ax_gender.set_title('Churn by Gender')
    st.pyplot(fig_gender)

    # Example 4: Age Distribution for Churned vs Retained
    fig_age, ax_age = plt.subplots(figsize=(8, 5))
    sns.boxplot(y='Age', x = 'Exited', hue = 'Exited',data = original_df , ax=ax_age)
    ax_age.set_title('Age Distribution by Churn Status')
    st.pyplot(fig_age)


except FileNotFoundError:
    st.sidebar.warning(f"Original data file not found at '{original_data_path}'. EDA plots based on the original dataset cannot be displayed.")
except Exception as e:
    st.sidebar.error(f"Error loading original data or generating EDA plots: {e}")


# --- About Section ---
st.sidebar.header("About")
st.sidebar.info("This app uses a machine learning model (Random Forest) to predict customer churn based on various attributes.")
st.sidebar.write("Developed by [Your Name]")
