import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# -------------------------------
# Load saved artifacts
# -------------------------------
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -------------------------------
# Define model (same as training)
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# Load model
model = MLP(len(feature_columns))
model.load_state_dict(torch.load("churn_model.pth"))
model.eval()

# -------------------------------
# UI
# -------------------------------
st.title("💳 Credit Card Churn Predictor")

st.write("Enter customer details to predict churn risk")

# Example inputs (you can expand these)
age = st.slider("Customer Age", 20, 80, 40)
credit_limit = st.number_input("Credit Limit", 1000, 50000, 10000)
total_trans_ct = st.number_input("Total Transactions", 0, 200, 50)
months_on_book = st.number_input("Months on Book", 1, 60, 36)
inactive_months = st.slider("Inactive Months (Last 12)", 0, 6, 2)

# -------------------------------
# Feature Engineering (same as training)
# -------------------------------
utilization_ratio = 0.5  # default placeholder
activity_rate = total_trans_ct / months_on_book if months_on_book > 0 else 0
high_inactivity = 1 if inactive_months > 2 else 0

# -------------------------------
# Build input dataframe
# -------------------------------
input_dict = {col: 0 for col in feature_columns}

# Fill known features
if "Customer_Age" in input_dict:
    input_dict["Customer_Age"] = age
if "Credit_Limit" in input_dict:
    input_dict["Credit_Limit"] = credit_limit
if "Total_Trans_Ct" in input_dict:
    input_dict["Total_Trans_Ct"] = total_trans_ct
if "Months_on_book" in input_dict:
    input_dict["Months_on_book"] = months_on_book
if "Months_Inactive_12_mon" in input_dict:
    input_dict["Months_Inactive_12_mon"] = inactive_months
if "Utilization_Ratio" in input_dict:
    input_dict["Utilization_Ratio"] = utilization_ratio
if "Activity_Rate" in input_dict:
    input_dict["Activity_Rate"] = activity_rate
if "High_Inactivity" in input_dict:
    input_dict["High_Inactivity"] = high_inactivity

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale
input_scaled = scaler.transform(input_df)

# Convert to tensor
input_tensor = torch.FloatTensor(input_scaled)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):
    with torch.no_grad():
        prediction = model(input_tensor)
        prob = prediction.item()
        
        if prob > 0.5:
            st.error(f"⚠️ High Risk of Churn ({prob:.2f})")
        else:
            st.success(f"✅ Low Risk of Churn ({prob:.2f})")

    st.write("### Interpretation")
    st.write("""
    - Higher probability → customer likely to leave
    - Lower probability → customer likely to stay
    """)
