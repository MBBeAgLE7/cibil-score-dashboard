import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pymongo import MongoClient
from fpdf import FPDF
import base64

# Connect to MongoDB
client = MongoClient("mongodb+srv://yash10nikam:77uGUmzGja0mDB0K@cluster0.ro59v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["myDatabase"]
collection = db["users"]

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

# Sidebar - User Input
st.sidebar.title("🔹 User Input")
user_id = st.sidebar.text_input("Enter User ID")

@st.cache_data
def fetch_user_data(user_id):
    user_data = collection.find_one({"Z": user_id}, {"_id": 0})
    return pd.DataFrame([user_data])[FEATURE_COLUMNS] if user_data else None

# Function to determine loan eligibility
def get_loan_eligibility(cibil_score):
    if cibil_score >= 750:
        return "Eligible for Home Loan, Car Loan, Personal Loan, Business Loan"
    elif 500 <= cibil_score < 750:
        return "Eligible for Personal Loan, Small Business Loan"
    else:
        return "Not eligible for major loans. Consider improving your credit score first."

# Function to provide improvement suggestions
def get_cibil_improvement_suggestions():
    return [
        "Pay your bills on time to improve your payment history.",
        "Reduce your debt-to-income ratio by lowering outstanding loans.",
        "Avoid multiple loan applications in a short period.",
        "Increase your savings and maintain a stable financial history.",
        "Use a mix of credit types (e.g., secured and unsecured loans)."
    ]

# Function to generate PDF report
def generate_pdf(user_id, cibil_score, risk_level, loan_eligibility, improvement_suggestions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Credit Score Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"User ID: {user_id}", ln=True)
    pdf.cell(200, 10, f"Predicted CIBIL Score: {cibil_score:.2f}", ln=True)

    # Replace emojis in `risk_level`
    risk_level_text = risk_level.replace("🟢", "(Low Risk)").replace("🟡", "(Medium Risk)").replace("🔴", "(High Risk)")
    pdf.cell(200, 10, f"Risk Level: {risk_level_text}", ln=True)

    pdf.cell(200, 10, f"Loan Eligibility: {loan_eligibility}", ln=True)
    pdf.ln(10)  # Line break
    pdf.cell(200, 10, "Recommendations to Improve CIBIL Score:", ln=True)

    for suggestion in improvement_suggestions:
        pdf.multi_cell(0, 10, f"- {suggestion}")

    pdf_filename = f"{user_id}_credit_report.pdf"
    pdf.output(pdf_filename, "F")

    return pdf_filename


# Function to create a downloadable link for the PDF
def get_pdf_download_link(pdf_filename):
    with open(pdf_filename, "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_filename}">📥 Download Report</a>'
    return href

# Main Content
st.title("📊 AI-Powered Credit Score Dashboard")

if user_id:
    user_data = fetch_user_data(user_id)

    if user_data is not None:
        # Predict CIBIL Score
        cibil_score = model.predict(user_data)[0]  

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

        # Get Loan Eligibility and Improvement Suggestions
        loan_eligibility = get_loan_eligibility(cibil_score)
        improvement_suggestions = get_cibil_improvement_suggestions()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"<p class='big-font'>Predicted CIBIL Score: <span style='color:cyan;'>{cibil_score:.2f}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<div class='risk-box' style='background-color:{risk_color}; color:white;'>{risk_level}</div>", unsafe_allow_html=True)

            st.subheader("🏦 Loan Eligibility")
            st.write(loan_eligibility)

            st.subheader("📌 How to Improve Your CIBIL Score")
            for suggestion in improvement_suggestions:
                st.markdown(f"- {suggestion}")

            # Generate PDF Report
            pdf_filename = generate_pdf(user_id, cibil_score, risk_level, loan_eligibility, improvement_suggestions)
            st.markdown(get_pdf_download_link(pdf_filename), unsafe_allow_html=True)

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
