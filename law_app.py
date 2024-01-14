import pandas as pd
import streamlit as st
from langchain import OpenAI

# from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import pinecone
import openai

# Initialize Pinecone and OpenAI


PINECONE_ENVIRONMENT = "gcp-starter"
INDEX_NAME = "law-gpt"

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT,  # next to api key in console
)

index = pinecone.Index(INDEX_NAME)


# Function to load data
@st.cache_data
def load_data():
    data = pd.read_csv("processed_data.csv")
    return data


# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def batch_embeddings(texts, batch_size=10):
    for i in range(0, len(texts), batch_size):
        response = openai.Embedding.create(
            input=texts[i : i + batch_size],
            engine="text-embedding-ada-002",
            api_key=OPENAI_API_KEY,
        )
        embeddings = [item["embedding"] for item in response["data"]]
        yield embeddings


def query_and_process_results(query_text, min_text_length=1, top_k=3):
    """
    Query the Pinecone index with the given text and process the results.

    Args:
    query_text (str): The text to query.
    min_text_length (int): Minimum length of text to include in results.
    top_k (int): Number of top results to retrieve.

    Returns:
    None
    """
    # Generate the query vector
    query_vector = next(batch_embeddings([query_text]))[0]

    # Perform the query and request metadata
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


def generate_summary(txt):
    # Instantiate the LLM model
    llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )

    docs = text_splitter.create_documents([txt])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=False
        #                                      verbose=True # Set verbose=True if you want to see the prompts being used
    )
    output = summary_chain.run(docs)
    return output


# Load the data
df = load_data()

# Streamlit app setup
st.title("Legal Cases Viewer")

# Display the dataframe in the app
st.write("Below are the legal cases:")
st.dataframe(df[["id", "name", "decision_date", "court_name"]])

# Streamlit app - Search functionality
st.write("## Search for Similar Cases")
search_query = st.text_input("Enter a search query:")
top_n = st.slider(
    "Select the number of top matches to display:", 1, 10, 3
)  # Add a slider for selecting top n matches

if st.button("Search"):
    if search_query:
        search_results = query_and_process_results(search_query, top_k=top_n)
        for result in search_results:
            st.markdown(f"#### ID: {result['id']}, Score: {result['score']}")
            st.write(result["text"])
            st.markdown("---")

# Streamlit app - Generate Case Summary with Original Text Option
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
