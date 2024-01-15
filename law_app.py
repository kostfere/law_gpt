import streamlit as st
import pandas as pd

import os

import time
from utils import (
    query_and_process_results,
    generate_summary,
    batch_embeddings,
    load_data,
)


@st.cache_data
def load_data():
    data = pd.read_csv("processed_data.csv")
    return data


df = load_data()

st.title("Legal Cases Viewer")


st.write("## Search for Similar Cases")
search_query = st.text_input("Enter a search query:")
top_n = st.slider("Select the number of top matches to display:", 1, 10, 3)

if st.button("Search"):
    if search_query:
        search_results = query_and_process_results(search_query, top_k=top_n)
        for result in search_results:
            st.markdown(f"#### ID: {result['id']}, Score: {result['score']}")
            st.write(result["text"])
            st.markdown("---")

st.write("## Generate Case Summary")
case_id = st.selectbox("Select a Case ID for summary:", df["id"])
show_original = st.checkbox("Show Original Text")


if st.button("Generate Summary"):
    selected_case_text = df[df["id"] == case_id]["text"].iloc[0]
    summary = generate_summary(selected_case_text)
    st.write("Summary:")
    st.write(summary)

    if show_original:
        st.write("Original Text:")
        st.write(selected_case_text)
