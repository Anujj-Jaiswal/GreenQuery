# === Imports ===
import os
import pandas as pd
import streamlit as st
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import pymysql
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from langchain_groq.chat_models import ChatGroq
from langchain_google_genai import GoogleGenerativeAI

# === Load Environment Variables ===
load_dotenv()

# === Custom Streamlit Response Parser ===
class StreamlitResponse(ResponseParser):
    def _init(self, context):
        super()._init_(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"], use_container_width=True)

    def format_plot(self, result):
        plot_val = result["value"]
        try:
            import matplotlib.pyplot as plt
            if hasattr(plot_val, "savefig"):
                st.pyplot(plot_val)
                return
        except ImportError:
            pass
        try:
            import plotly.graph_objs as go
            if isinstance(plot_val, go.Figure):
                st.plotly_chart(plot_val, use_container_width=True)
                return
        except ImportError:
            pass
        st.image(plot_val)

    def format_other(self, result):
        st.write(result["value"])

# === Sidebar UI ===
st.sidebar.title("Settings")

# Model selection
model_choice = st.sidebar.selectbox("Choose LLM engine", ["Groq (LLaMA)", "Gemini (Google)"])

# Data source selection
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV", "MySQL Database"])
uploaded_file = None
connect_db = False

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("📂 Upload Records", type="csv")
elif data_source == "MySQL Database":
    st.sidebar.text_input("Host", key="mysql_host", placeholder="e.g., localhost")
    st.sidebar.text_input("User ", key="mysql_user", placeholder="e.g., root")
    st.sidebar.text_input("Password", key="mysql_password", type="password")
    st.sidebar.text_input("Database", key="mysql_database", placeholder="e.g., my_database")
    st.sidebar.text_input("Table Name", key="mysql_table", placeholder="e.g., my_table")
    connect_db = st.sidebar.button("Connect to Database")

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.pop("data", None)
    st.session_state.pop("data_source_type", None)

# === Initialize Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === App Title ===
st.title("GreenQuery🌿")

# === LLM Initialization ===
llm = None
if model_choice == "Gemini (Google)":
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        st.warning("⚠ Google API key not found.")
    else:
        try:
            llm = GoogleGenerativeAI(api_key=google_key, model="gemini-2.0-flash")
            st.success("gemini-2.0-flash loaded.")
        except Exception as e:
            st.error(f"Gemini model error: {e}")

elif model_choice == "Groq (LLaMA)":
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.warning("⚠ Groq API key not found.")
    else:
        try:
            llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=groq_key)
            st.success("deepseek-r1-distill-llama-70b loaded.")
        except Exception as e:
            st.error(f"Groq model error: {e}")

# === Load Data ===
data = None
if data_source == "Upload CSV" and uploaded_file and llm:
    try:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.session_state.data_source_type = "CSV"
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()

elif data_source == "MySQL Database" and connect_db and llm:
    try:
        mysql_host = st.session_state.mysql_host
        mysql_user = st.session_state.mysql_user
        mysql_password = st.session_state.mysql_password
        mysql_database = st.session_state.mysql_database
        mysql_table = st.session_state.mysql_table

        if not all([mysql_host, mysql_user, mysql_password, mysql_database, mysql_table]):
            st.error("❌ Please fill in all MySQL connection details.")
            st.stop()

        connection = pymysql.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database
        )
        query = f"SELECT * FROM {mysql_table}"
        data = pd.read_sql(query, con=connection)
        connection.close()
        st.session_state.data = data
        st.session_state.data_source_type = "MySQL"
    except Exception as e:
        st.error(f"❌ Failed to connect to MySQL database: {e}")
        st.stop()

# === Display Data Preview First ===
smart_df = None
if "data" in st.session_state and llm:
    data = st.session_state.data

    # Optional search
    search_query = st.text_input("Search in DataFrame", placeholder="Type to filter rows...")
    if search_query:
        filtered_data = data[data.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]
        st.subheader("Filtered Data Preview")
        st.dataframe(filtered_data, use_container_width=True)
    else:
        st.subheader("Data Preview")
        st.dataframe(data, use_container_width=True)

    # === Threshold Alerts Feature ===
    st.subheader("⚠️ Threshold Alerts and Suggestions")

    thresholds = {
        "laptop_hours": {
            "threshold": 8.0,
            "message": "Usage exceeds 8 hours — might indicate prolonged idle or unnecessary laptop activity. Consider encouraging breaks or powering down devices when idle."
        },
        "ac_power_consumed": {
            "threshold": 2.5,
            "message": "High AC power consumption detected. Evaluate temperature settings or optimize AC usage schedules to reduce energy waste."
        },
        "projector_power_consumed": {
            "threshold": 1.0,
            "message": "Projector usage over 1 kWh detected. Consider limiting projector use during long meetings or ensuring it is turned off when not needed."
        },
        "energy_consumed": {
            "threshold": 3.5,
            "message": "Overall energy consumption is above typical levels. Recommend auditing individual energy use and promoting energy-saving practices."
        },
        "co2_emission": {
            "threshold": 1.5,
            "message": "Elevated CO2 emissions detected. Encourage sustainable practices or review equipment efficiency to reduce environmental impact."
        },
    }

    alert_found = False
    for col, info in thresholds.items():
        if col in data.columns:
            exceeding_rows = data[data[col] > info["threshold"]]
            if not exceeding_rows.empty:
                alert_found = True
                st.markdown(f"### Alerts for **{col}** (Threshold: {info['threshold']})")
                for idx, row in exceeding_rows.iterrows():
                    identifier = f"Employee: {row.get('employee_name', 'N/A')} (ID: {row.get('employee_id', 'N/A')})"
                    value = row[col]
                    st.markdown(f"- **{identifier}** has **{col} = {value}**, which exceeds the threshold.")
                    st.markdown(f"  > Suggestion: {info['message']}")
                st.markdown("---")

    if not alert_found:
        st.info("No threshold breaches detected in the current dataset.")

    # === Setup SmartDataframe ===
    try:
        smart_df = SmartDataframe(data, config={
            "llm": llm,
            "response_parser": StreamlitResponse,
            "enforce_privacy": False
        })
        st.success("SmartDataframe is ready!")
    except Exception as e:
        st.error(f"Error initializing SmartDataframe: {e}")
        st.stop()

    # === Chat History Replay ===
    if st.session_state.chat_history:
        st.subheader("📂 Previous Queries")
        for i, query in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(f"*Query {i+1}:* {query}")
            with st.chat_message("assistant"):
                try:
                    result = smart_df.chat(query)
                    if isinstance(result, dict) and "value" in result:
                        val = result["value"]
                        if isinstance(val, pd.DataFrame):
                            st.dataframe(val, use_container_width=True)
                        elif isinstance(val, (str, int, float)):
                            st.write(val)
                        else:
                            st.write("Unsupported response type.")
                    else:
                        st.write(result)
                except Exception as e:
                    st.error(f"Error rerunning query {i + 1}: {e}")

    # === Query Section ===
    st.divider()
    st.subheader("🔍 Ask a Question")

    query = st.text_area("Enter your data query", placeholder="e.g. Top 5 contributors by CO2 emission")

    if st.button("Generate Response") and query:
        with st.spinner("Generating response..."):
            try:
                st.session_state.chat_history.append(query)
                result = smart_df.chat(query)
                if result:
                    st.write(result)
            except Exception as e:
                st.error(f"Error generating response: {e}")

else:
    st.info("Upload records or connect to a database and select a model to get started.")

st.markdown("""
    <style>
    .bottom-right {
        position: fixed;
        bottom: 10px;
        right: 20px;
        font-size: 1.1em;
        color: white;
        z-index: 9999;
    }
    .bottom-right a {
        color: white;
        text-decoration: none;
        font-weight: bold;
        font-size: 1.5em;
        
    }
    .bottom-right a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="bottom-right">
        <a href="https://linktr.ee/_8ball?utm_source=linktree_profile_share<sid=a9f65566-6202-4681-a988-5c9e24985696" target="_blank">
            synapse
        </a>
    </div>
""", unsafe_allow_html=True)