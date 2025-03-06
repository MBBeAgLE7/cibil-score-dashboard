import streamlit as st
import pandas as pd
import joblib
import pymongo
import plotly.graph_objects as go
from pymongo import MongoClient

# Connect to MongoDB
df = pd.read_csv('credit_score.csv')
client = MongoClient("mongodb+srv://yash10nikam:<77uGUmzGja0mDB0K>@cluster0.ro59v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB URI
db = client["credit_db"]
collection = db["credit_scores"]

collection.insert_many(df.to_dict("records"))