import os
import pandas as pd
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, skew, kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define file paths
weather_dir = r"D:\dataScience-Assignments\raw\weather_raw_data"
electricity_dir = r"D:\dataScience-Assignments\raw\electricity_raw_data"
output_file = r"D:\dataScience-Assignments\final_cleaned_data.csv"

# Load Weather Data
def load_weather_data(weather_dir):
    weather_data = []
    
    for file in glob.glob(os.path.join(weather_dir, "*.csv")):
        df = pd.read_csv(file, parse_dates=['date'])
        df.rename(columns={'date': 'timestamp', 'temperature_2m': 'temperature'}, inplace=True)
        weather_data.append(df)

    return pd.concat(weather_data, ignore_index=True) if weather_data else pd.DataFrame()

# Load Electricity Demand Data
def load_electricity_data(electricity_dir):
    electricity_data = []
    
    for file in glob.glob(os.path.join(electricity_dir, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "response" in data and "data" in data["response"]:
            for entry in data["response"]["data"]:
                electricity_data.append({
                    "timestamp": entry["period"],
                    "zone": entry["subba-name"],
                    "electricity_demand": float(entry["value"])
                })

    df = pd.DataFrame(electricity_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%dT%H")
    
    return df

# Data Preprocessing
def preprocess_data(weather_df, electricity_df):
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], utc=True).dt.tz_convert(None)
    electricity_df['timestamp'] = pd.to_datetime(electricity_df['timestamp'], utc=True).dt.tz_convert(None)

    merged_df = pd.merge(weather_df, electricity_df, on='timestamp', how='inner')

    # Handle Missing Values
    missing_percent = merged_df.isnull().mean() * 100
    print("Missing Data Percentage:\n", missing_percent)

    merged_df.ffill(inplace=True)  

    # Remove Duplicates
    merged_df.drop_duplicates(inplace=True)

    # Feature Engineering
    merged_df['hour'] = merged_df['timestamp'].dt.hour
    merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    merged_df['month'] = merged_df['timestamp'].dt.month

    # Drop timestamp if unnecessary
    merged_df.drop(columns=['timestamp'], inplace=True)

    return merged_df

def perform_eda(df):
    #  Convert all numeric columns to float
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].astype(float)

    print("\nStatistical Summary:\n", df[numeric_cols].describe())
    print("\nSkewness:\n", df[numeric_cols].apply(skew))
    print("\nKurtosis:\n", df[numeric_cols].apply(kurtosis))

    # Correlation Heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Time Series Decomposition
    result = seasonal_decompose(df['electricity_demand'], model='additive', period=24)
    result.plot()
    plt.show()

    # Stationarity Test
    adf_test = adfuller(df['electricity_demand'])
    print(f"ADF Statistic: {adf_test[0]}")
    print(f"p-value: {adf_test[1]}")
    print("Stationary" if adf_test[1] < 0.05 else "Non-Stationary")


# Outlier Detection
def detect_outliers(df):
    df['z_score'] = zscore(df['electricity_demand'])
    df['outlier_z'] = df['z_score'].abs() > 3

    Q1 = df['electricity_demand'].quantile(0.25)
    Q3 = df['electricity_demand'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df['outlier_iqr'] = (df['electricity_demand'] < lower_bound) | (df['electricity_demand'] > upper_bound)

    return df

# Regression Model
def train_regression(df):
    features = ['hour', 'day_of_week', 'month', 'temperature']
    X = df[features]
    y = df['electricity_demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"R² Score: {r2_score(y_test, y_pred)}")

# Load data
weather_df = load_weather_data(weather_dir)
electricity_df = load_electricity_data(electricity_dir)

if not weather_df.empty and not electricity_df.empty:
    final_df = preprocess_data(weather_df, electricity_df)
    final_df.to_csv(output_file, index=False)
    print(f" Processed data saved at {output_file}")

    perform_eda(final_df)
    final_df = detect_outliers(final_df)
    train_regression(final_df)

else:
    print(" No data processed. Check input directories.")
