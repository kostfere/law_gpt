import streamlit as st
import pandas as pd
import openai
import pinecone
import os
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Initialize Pinecone and OpenAI settings
PINECONE_ENVIRONMENT = "gcp-starter"
INDEX_NAME = "law-gpt"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)


# Load data function
@st.cache_data
def load_data():
    data = pd.read_csv("processed_data.csv")
    return data


# Streamlit app setup
st.title("Legal Cases Viewer")

# OpenAI API Key Management
st.sidebar.header("API Key Configuration")
API_KEY_ENV = os.getenv("API_KEY")  # Loaded from environment
key_choice = st.sidebar.radio(
    "Choose your API Key source:",
    ("Use Your Own Key", "Use Free Key (capped)"),
    horizontal=True,
)

if key_choice == "Use Your Own Key":
    API_Key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
elif key_choice == "Use Free Key (capped)":
    API_Key = API_KEY_ENV

# Display message based on API key selection
if not API_Key:
    st.sidebar.warning("API Key is required to use the app's features.")
else:
    st.sidebar.success("API Key loaded successfully.")

# Initialize OpenAI client with the selected API Key
openai_client = OpenAI(api_key=API_Key)


# Function to generate embeddings batch-wise
def batch_embeddings(texts, batch_size=10):
    for i in range(0, len(texts), batch_size):
        response = openai.Embedding.create(
            input=texts[i : i + batch_size],
            engine="text-embedding-ada-002",
            api_key=API_Key,
        )
        embeddings = [item["embedding"] for item in response["data"]]
        yield embeddings


# Query Pinecone and process results
def query_and_process_results(query_text, min_text_length=1, top_k=3):
    query_vector = next(batch_embeddings([query_text]))[0]
    query_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    results = []
    for result in query_results["matches"]:
        if (
            "metadata" in result
            and "text" in result["metadata"]
            and len(result["metadata"]["text"]) > min_text_length
        ):
            results.append(
                {
                    "id": result["id"],
                    "score": result["score"],
                    "text": result["metadata"]["text"],
                }
            )
    return results


# Function to generate summary
def generate_summary(txt):
    llm = OpenAI(api_key=API_Key, temperature=0)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([txt])
    summary_chain = load_summarize_chain(
        llm=llm, chain_type="map_reduce", verbose=False
    )
    output = summary_chain.run(docs)
    return output


# Load data and display in app
df = load_data()
st.write("Below are the legal cases:")
st.dataframe(df[["id", "name", "decision_date", "court_name"]])

# Search functionality
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
