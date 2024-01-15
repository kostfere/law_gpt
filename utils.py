import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import langchain_openai
import os
import pandas as pd
from openai import OpenAI
import pinecone
from tqdm import tqdm
import time
from typing import Callable, List, Any


@st.cache_data
def load_data():
    data = pd.read_csv("processed_data.csv")
    return data


def create_embedding(text, client, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def batch_embeddings(texts, client, batch_size=10, engine="text-embedding-ada-002"):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        for text in batch_texts:
            embedding = create_embedding(text, client, engine)
            all_embeddings.append(embedding)
    return all_embeddings


def query_and_process_results(query_text, index, client, top_k=3, retries=3):
    for attempt in range(retries):
        try:
            query_vector = create_embedding(query_text, client)
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


def generate_summary(txt, api_key):
    llm = langchain_openai.OpenAI(api_key=api_key, temperature=0)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([txt])
    summary_chain = load_summarize_chain(
        llm=llm, chain_type="map_reduce", verbose=False
    )
    output = summary_chain.run(docs)
    return output

def upload_embeddings_to_pinecone(
    df: pd.DataFrame, 
    text_splitter: Any, 
    batch_embeddings: Callable[[List[str], Any], List[List[float]]], 
    client: Any, 
    index: Any, 
    upload_threshold: int = 5
) -> None:
    """Upload embeddings to Pinecone in batches.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_splitter (Any): Function to split text into smaller chunks.
        batch_embeddings (Callable): Function to generate embeddings for text chunks.
        client (Any): Client object for the embeddings service.
        index (Any): Pinecone index object.
        upload_threshold (int): Number of rows to process before uploading.
    """
    data_to_upload = []
    processed_rows = 0

    for _, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
        chunks = text_splitter.split_text(row["text"])
        for chunk_index, chunk in enumerate(chunks):
            chunk_embeddings = batch_embeddings([chunk], client)
            # Assuming the first item in the list is the embedding you need
            chunk_embedding = chunk_embeddings[0]
            metadata = {"text": chunk, "original_id": row["id"]}
            data_to_upload.append(
                (f"{row['id']}-{chunk_index}", chunk_embedding, metadata)
            )

        processed_rows += 1
        if processed_rows >= upload_threshold:
            index.upsert(vectors=data_to_upload)
            data_to_upload = []
            processed_rows = 0

    # Upload any remaining data
    if data_to_upload:
        index.upsert(vectors=data_to_upload)

    
def query_legal_case(query, index, llm, top_k=3):
    """
    Query specific legal case details based on the query.

    Args:
    query (str): The query or case ID to be searched.
    index (pinecone.Index): Pinecone index for querying vectors.
    llm (langchain.ChatOpenAI): Initialized LangChain's ChatOpenAI model.
    top_k (int): Number of top results to retrieve.

    Returns:
    list: List of responses from the QA chain.
    """
    # Perform similarity search using Pinecone
    query_vector = create_embedding(query, client)
    query_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # Extract texts from query results
    docs = [result["metadata"]["text"] for result in query_results["matches"]]

    # Load QA Chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Run QA Chain
    return chain.run(input_documents=docs, question=query)
