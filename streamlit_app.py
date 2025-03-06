import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb+srv://yash10nikam:<77uGUmzGja0mDB0K>@cluster0.ro59v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB URI
db = client["credit_db"]
collection = db["credit_scores"]

# Load the trained model
model = joblib.load("credit_score_model.pkl")

# Define feature columns
FEATURE_COLUMNS = [
    "INCOME", "SAVINGS", "DEBT", "R_SAVINGS_INCOME", "R_DEBT_INCOME",
    "R_DEBT_SAVINGS", "R_EDUCATION_INCOME", "R_EXPENDITURE_SAVINGS",
    "R_EXPENDITURE_INCOME", "CAT_MORTGAGE", "R_ENTERTAINMENT_INCOME", 
    "R_GROCERIES_INCOME"
]

# Page Configuration
st.set_page_config(page_title="AI Credit Score Dashboard", layout="wide")

# Custom CSS for modern UI
st.markdown("""
    <style>
        body { background-color: #0e1117; color: white; }
        .big-font { font-size: 24px !important; font-weight: bold; }
        .risk-box { padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Sidebar - User Input
st.sidebar.title("🔹 User Input")
user_id = st.sidebar.text_input("Enter User ID")

@st.cache_data
def fetch_user_data(user_id):
    user_data = collection.find_one({"Z": user_id}, {"_id": 0})  # Exclude MongoDB _id field
    return pd.DataFrame([user_data])[FEATURE_COLUMNS] if user_data else None

# Main Content
st.title("📊 AI-Powered Credit Score Dashboard")

if user_id:
    user_data = fetch_user_data(user_id)

    if user_data is not None:
        # Layout: Two columns
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🔍 Fetched Financial Data")
            st.dataframe(user_data.style.set_properties(**{'background-color': '#262730', 'color': 'white'}))

            # Predict CIBIL Score
            cibil_score = model.predict(user_data)[0]  
            st.markdown(f"<p class='big-font'>Predicted CIBIL Score: <span style='color:cyan;'>{cibil_score:.2f}</span></p>", unsafe_allow_html=True)

            # Determine risk level
            if cibil_score >= 750:
                risk_level = "Low Risk 🟢"
                risk_color = "green"
            elif 500 <= cibil_score < 750:
                risk_level = "Medium Risk 🟡"
                risk_color = "yellow"
            else:
                risk_level = "High Risk 🔴"
                risk_color = "red"

            st.markdown(f"<div class='risk-box' style='background-color:{risk_color}; color:white;'>{risk_level}</div>", unsafe_allow_html=True)

        with col2:
            # Financial Trends Visualization
            income = user_data["INCOME"].values[0]
            savings = user_data["SAVINGS"].values[0]
            debt = user_data["DEBT"].values[0]
            categories = ["Income", "Savings", "Debt"]
            values = [income, savings, debt]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=categories, 
                y=values, 
                mode='lines+markers',
                marker=dict(size=12, color='cyan', symbol='circle'),
                line=dict(color='cyan', width=3),
                hoverinfo='text',
                text=[f"{cat}: {val}" for cat, val in zip(categories, values)]
            ))

            fig.update_layout(
                title="📈 Financial Overview",
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white"),
                xaxis=dict(title="Category", color="white"),
                yaxis=dict(title="Amount", color="white"),
                margin=dict(l=40, r=40, t=40, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("User ID not found.")