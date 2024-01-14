import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os
import openai
import pinecone
from tqdm import tqdm
from dotenv import load_dotenv
import time

# from utils import (query_and_process_results, generate_summary, batch_embeddings, load_data)


load_dotenv()

# Initialize Pinecone and OpenAI settings
PINECONE_ENVIRONMENT = "gcp-starter"
INDEX_NAME = "law-gpt"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)

@st.cache_data
def load_data():
    data = pd.read_csv("processed_data.csv")
    return data


# Function to generate embeddings batch-wise


def create_embedding(text, engine="text-embedding-ada-002", api_key=OPENAI_API_KEY):
    response = openai.Embedding.create(
        input=[text],  # Wrap the text in a list
        engine=engine,
        api_key=api_key,
    )
    # Assuming response is successful and contains data
    embedding = response["data"][0]["embedding"]
    return embedding


def batch_embeddings(
    texts, batch_size=10, engine="text-embedding-ada-002", api_key=OPENAI_API_KEY
):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        for text in batch_texts:
            embedding = create_embedding(text, engine, api_key)
            all_embeddings.append(embedding)
    return all_embeddings


def query_and_process_results(query_text, top_k=3, retries=3):
    for attempt in range(retries):
        try:
            query_vector = create_embedding(query_text)
            query_results = index.query(
                vector=query_vector, top_k=top_k, include_metadata=True
            )
            results = []
            for result in query_results["matches"]:
                if "metadata" in result and "text" in result["metadata"]:
                    results.append(
                        {
                            "id": result["id"],
                            "score": result["score"],
                            "text": result["metadata"]["text"],
                        }
                    )
            return results
        except Exception as e:
            if attempt < retries - 1:  # if it's not the last attempt
                time.sleep(2)  # wait for 2 seconds before retrying
                continue
            else:
                raise e


def generate_summary(txt, api_key=OPENAI_API_KEY):
    llm = langchain.OpenAI(api_key=api_key, temperature=0)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([txt])
    summary_chain = load_summarize_chain(
        llm=llm, chain_type="map_reduce", verbose=False
    )
    output = summary_chain.run(docs)
    return output

def upload_embeddings_to_pinecone(df, text_splitter, batch_embeddings, index, upload_threshold=5):
    """Upload embeddings to Pinecone in batches.
    
    Args:
    df: DataFrame containing the text data.
    text_splitter: Function to split text into smaller chunks.
    batch_embeddings: Function to generate embeddings for text chunks.
    index: Pinecone index object.
    upload_threshold: Number of rows to process before uploading.
    """
    data_to_upload = []
    processed_rows = 0

    for _, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
        chunks = text_splitter.split_text(row['text'])
        for chunk_index, chunk in enumerate(chunks):
            chunk_embedding = next(batch_embeddings([chunk]))
            metadata = {'text': chunk, 'original_id': row['id']}
            data_to_upload.append((f"{row['id']}-{chunk_index}", chunk_embedding[0], metadata))
        
        processed_rows += 1
        if processed_rows >= upload_threshold:
            index.upsert(vectors=data_to_upload)
            data_to_upload = []
            processed_rows = 0

    # Upload any remaining data
    if data_to_upload:
        index.upsert(vectors=data_to_upload)

    # upload_embeddings_to_pinecone(df, text_splitter, batch_embeddings, index)

