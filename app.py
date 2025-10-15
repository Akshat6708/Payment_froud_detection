import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Payment Fraud Detection Dashboard")

# Load your trained model (update path if needed)
model = joblib.load("XBC_model.pkl")

def preprocess_data(df):
    required_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']
    # Drop all columns except required
    df = df[[col for col in required_columns if col in df.columns]]
    # Label encode 'type'
    if 'type' in df.columns:
        le = LabelEncoder()
        df['type'] = le.fit_transform(df['type'])
    return df

uploaded_file = st.file_uploader("Upload CSV file for fraud prediction", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Original Data Preview")
    st.write(df.head())

    # Preprocess CSV
    processed_df = preprocess_data(df)
    st.subheader("Processed Data for Model")
    st.write(processed_df.head())
    
    # Run Prediction
    prediction = model.predict(processed_df)
    processed_df["Prediction"] = prediction
    st.subheader("Prediction Results")
    st.write(processed_df.head(200))

    # Graph: Bar chart for prediction distribution
    st.subheader("Fraud Prediction Distribution")
    st.bar_chart(processed_df["Prediction"].value_counts())


    # Optional: If you want a matplotlib chart
    fig, ax = plt.subplots()
    processed_df["Prediction"].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    st.pyplot(fig)
