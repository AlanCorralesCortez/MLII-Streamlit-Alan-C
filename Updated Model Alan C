import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

upgrade pip

# Function to convert currency to float
def currency_to_float(currency_val):
    if isinstance(currency_val, str):
        return float(currency_val.replace(',', '').replace('$', ''))
    return currency_val

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Convert currency fields from string to numeric
    currency_columns = ['Unit Cost', 'Unit Price', 'Total Sales']
    for col in currency_columns:
        data[col] = data[col].apply(currency_to_float)

    # Convert date fields to datetime
    date_columns = ['ProcuredDate', 'OrderDate', 'ShipDate', 'DeliveryDate']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    # Feature Engineering - Example: Calculate the delivery lead time
    data['DeliveryLeadTime'] = (data['DeliveryDate'] - data['OrderDate']).dt.days

    # Drop the original date columns and any other non-relevant columns
    data.drop(columns=date_columns, inplace=True)

    # Drop any rows with missing target values
    data.dropna(subset=['Total Sales'], inplace=True)

    return data

# Set the file path (this path needs to be accessible by Streamlit)
file_path = 'C:/Users/alanc/Downloads/US_Regional_Sales_Data.csv'

# Load and preprocess the data
data = load_and_preprocess_data(file_path)

# Separating features and target
X = data.drop(columns=['Total Sales'])
y = data['Total Sales']

# Identifying categorical columns (after removing date columns)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Creating a ColumnTransformer for transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Initialize Streamlit app
st.title('Sales Data Analysis')

# Train and display Random Forest Model results
st.header('Random Forest Model')
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
st.write(f"Random Forest MSE: {mse_rf}")

# Feature importance from Random Forest
rf_model = rf_pipeline.named_steps['model']
ohe = (rf_pipeline.named_steps['preprocessor']
       .named_transformers_['cat']
       .get_feature_names_out(input_features=categorical_cols))
feature_names = np.r_[X.select_dtypes(include=['int64', 'float64']).columns, ohe]
importances = rf_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
sorted_feature_names = feature_names[sorted_indices]
sorted_importances = importances[sorted_indices]

# Plotting feature importances
st.subheader('Feature Importance')
fig, ax = plt.subplots()
ax.barh(sorted_feature_names[-10:], sorted_importances[-10:])  # Show top 10 features
ax.set_xlabel('Importance')
ax.set_title('Top 10 Feature Importances from Random Forest')
st.pyplot(fig)

# Train and display Linear Regression Model results
st.header('Linear Regression Model')
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', LinearRegression())])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
st.write(f"Linear Regression MSE: {mse_lr}")
st.write(f"Linear Regression R^2: {r2_lr}")

# Scatter plot for actual vs predicted values
st.subheader('Actual vs Predicted Values')
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_lr, alpha=0.3)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.title('Linear Regression Predictions')
st.pyplot(fig)
