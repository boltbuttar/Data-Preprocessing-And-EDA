
# **📊 Data Preprocessing And EDA**  
**📌 A Complete Data Pipeline for Electricity Demand Analysis**  

This project focuses on **data preprocessing, exploratory data analysis (EDA), outlier detection, and regression modeling** to analyze and predict **electricity demand** based on historical **weather conditions** and **time-based features**.  

🚀 **Key Highlights:**  
✅ **Loads & Merges** electricity demand (JSON) & weather data (CSV)  
✅ **Handles Missing Data** using forward-fill strategy  
✅ **Detects & Removes Outliers** using Z-score & IQR method  
✅ **Performs Statistical Analysis & Visualizations**  
✅ **Builds a Machine Learning Model** for electricity demand forecasting  

---

## **📂 Project Workflow**
### **1️⃣ Data Loading & Integration**  
📌 Reads **multiple JSON & CSV files** from the dataset directories  
📌 Merges weather and electricity demand data into a **single structured dataset**  

### **2️⃣ Data Cleaning & Preprocessing**  
📌 Converts **timestamps** to proper `datetime` format  
📌 Extracts **hour, day, and month** for time-based analysis  
📌 Handles **missing values** using forward-fill (`ffill()`)  

### **3️⃣ Exploratory Data Analysis (EDA)**  
📊 Computes **statistical summaries** (Mean, Std, Skewness, Kurtosis)  
📉 **Trend & Seasonality Analysis** using **Time Series Decomposition**  
🔥 **Correlation Heatmap** to visualize relationships between variables  

### **4️⃣ Outlier Detection & Handling**  
⚠️ Identifies **outliers** using **Z-score & IQR method**  
🔍 Removes or caps extreme values to maintain data integrity  

### **5️⃣ Machine Learning: Regression Model**  
🤖 Builds a **Linear Regression Model** to predict electricity demand based on:  
✔ **Hour of the Day**  
✔ **Day of the Week**  
✔ **Month**  
✔ **Temperature**  

📈 Evaluates the model using **MSE, RMSE, and R² Score**  

---

## **📂 File Structure**
```
📦 Data Preprocessing And EDA
 ┣ 📂 raw_data
 ┃ ┣ 📂 weather_raw_data  # CSV files
 ┃ ┣ 📂 electricity_raw_data  # JSON files
 ┣ 📜 electricity_demand_analysis.ipynb  # Jupyter Notebook (Step-by-step execution)
 ┣ 📜 electricity_demand_analysis.py  # Python script for full pipeline execution
 ┣ 📜 final_cleaned_data.csv  # Merged and cleaned dataset
 ┣ 📜 README.md  # Project Documentation
```

---

## **⚡ Getting Started**
### **🔹 Prerequisites**
Ensure you have the required Python libraries installed:
```sh
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels
```

### **🔹 Running the Project**
Run the full **Python script**:
```sh
python electricity_demand_analysis.py
```
Or open **Jupyter Notebook** to execute step-by-step:
```sh
jupyter notebook
```

---

## **🤝 Contributing**
🔹 Fork the repository, raise issues, and submit pull requests 🚀  
📩 **For inquiries, contact:** asgharfur92@gmail.com 

---

