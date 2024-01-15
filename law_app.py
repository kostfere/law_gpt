import streamlit as st
import pandas as pd
import pinecone
from openai import OpenAI
from langchain import hub

import time
from utils import (
    query_and_process_results,
    generate_summary,
    # batch_embeddings,
    case_qa,
    load_data,
)

PINECONE_ENVIRONMENT = "gcp-starter"
INDEX_NAME = "law-gpt"
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)
client = OpenAI(api_key=OPENAI_API_KEY)
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")


@st.cache_data
def load_data():
    data = pd.read_csv("processed_data.csv")
    return data


df = load_data()

st.title("Legal Cases Viewer")


st.write("## Search Cases")
search_query = st.text_input("Enter a search query:")
top_n = st.slider("Select the number of top matches to display:", 1, 10, 3)

if st.button("Search"):
    if search_query:
        search_results = query_and_process_results(
            search_query, index, client, top_k=top_n
        )
        for result in search_results:
            st.markdown(f"#### ID: {result['id']}, Score: {result['score']}")
            st.write(result["text"])
            st.markdown("---")

st.write("## Generate Case Summary")
case_id = st.selectbox("Select a Case ID for summary:", df["id"])
show_original = st.checkbox("Show Original Text")


if st.button("Generate Summary"):
    selected_case_text = df[df["id"] == case_id]["text"].iloc[0]
    summary = generate_summary(selected_case_text, OPENAI_API_KEY)
    st.write("Summary:")
    st.write(summary)

    if show_original:
        st.write("Original Text:")
        st.write(selected_case_text)

st.write("## Ask a Question about a Case")
selected_case_id = st.selectbox("Select a Case ID to ask a question about:", df["id"])
user_question = st.text_input("Enter your question about the case:")

if st.button("Get Answer"):
    if user_question:
        # Assuming you have a function to process the question and fetch the answer
        # Example: get_case_answer(case_id, question, index, client)
        answer = case_qa(selected_case_id, user_question, prompt, client, index)
        st.write("Answer:")
        st.write(answer)
