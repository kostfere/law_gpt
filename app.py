import streamlit as st
import pandas as pd

# Function to load data
@st.cache_data  # Updated caching decorator
def load_data():
    data = pd.read_csv('processed_data.csv')
    return data

# Load the data
df = load_data()

# Streamlit app
st.title('Legal Cases Viewer')

# Display the dataframe in the app
st.write("Below are the legal cases:")
st.dataframe(df)

# Optional: Add a selectbox to view details of a specific case
case_id = st.selectbox('Select a Case ID to view details:', df['id'])
selected_case = df[df['id'] == case_id]

if not selected_case.empty:
    st.write("Case Details:")
    st.json(selected_case.to_json(orient='records'))
 