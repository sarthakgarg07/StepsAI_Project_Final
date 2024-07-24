import os
import io
import time
import streamlit as st
import PyPDF2
import nltk
import json
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
import numpy as np
from sklearn.mixture import GaussianMixture
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Initialize the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5

# Download NLTK punkt tokenizer data
nltk.download('punkt')

# Initialize page
st.set_page_config(page_title="PDF Content Analyzer")
st.header("PDF Content Analyzer")
st.sidebar.title("Options")

# Clear messages
clear_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You are a helpful assistant specializing in content extraction and question answering.")
    ]
    st.session_state.costs = []

# Select LLM
llm = ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME, openai_api_key=openai.api_key)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ''
    with pdf_file as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to chunk text
def chunk_text(text, chunk_size=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += sentence + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to embed chunks
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

# Function to summarize text using GPT
def summarize_text_gpt(text):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error occurred."

# Recursive clustering and summarization
def recursive_clustering(embeddings, chunks, depth=2, current_depth=1):
    if current_depth > depth or len(chunks) < 2:
        return {f'level_{current_depth}': chunks}
    
    gmm = GaussianMixture(n_components=2, covariance_type='tied')
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[idx])

    summaries = {}
    for cluster_id, cluster_chunks in clusters.items():
        combined_text = ' '.join(cluster_chunks)
        summary = summarize_text_gpt(combined_text)
        summary_embedding = embed_chunks([summary])
        summaries[cluster_id] = recursive_clustering(summary_embedding, [summary], depth, current_depth + 1)

    return summaries

# Process each uploaded text
all_embeddings = []
texts = []
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if pdf_files:
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        pdf_name = pdf_file.name
        texts.append((pdf_name, text))
        st.sidebar.success(f"PDF text extracted successfully from {pdf_name}!")


    for pdf_name,text in texts:
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        hierarchical_structure = recursive_clustering(embeddings, chunks, 2)
        all_embeddings.append((embeddings, chunks, pdf_name, hierarchical_structure))
    
# Connect to MILVUS
def connect_to_milvus():
    try:
        connections.connect("default", host="0.0.0.0", port="19530")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),  # Adjust dim according to your embedding size
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, "index1")
        collection = Collection("exmpcollection1", schema)
        collection.create_index(field_name="embeddings", index_params={"index_type": "IVF_FLAT","params": {"nlist": 1024} ,"metric_type": "L2"})
        collection.load()
        return collection
    except Exception as e:
        error_message = str(e)
        if "Fail connecting to server" in error_message:
            st.error("Failed to connect to Milvus: Server is not running or connection parameters are incorrect.")
        elif "illegal connection params" in error_message:
            st.error("Failed to connect to Milvus: Illegal connection parameters.")
        else:
            st.error(f"Failed to connect to Milvus: {e}")
        return None

collection = connect_to_milvus()

if collection and all_embeddings:
    try:
        for embeddings, chunks, pdf_name, hierarchical_structure in all_embeddings:
            num_embeddings = embeddings.shape[0]
            metadata = [{"pdf_name": pdf_name, "chunk": chunk, "chunk_index": i, "hierarchical_level": 1} for i, chunk in enumerate(chunks)]
            
            if num_embeddings != len(metadata):
                st.error(f"Mismatch in embeddings and metadata length: {num_embeddings} vs {len(metadata)}")
                continue  # Skip this batch if there's a mismatch

            # Include hierarchical summaries
            for level, summaries in hierarchical_structure.items():
                if isinstance(level, str) and "level_" in level:
                    level_num = int(level.split('_')[1])
                    for summary in summaries:
                        summary_embedding = embed_chunks([summary])[0]
                        embeddings = np.vstack([embeddings, summary_embedding])
                        metadata.append({"pdf_name": pdf_name, "chunk": summary, "chunk_index": -1, "hierarchical_level": level_num})
            
            # Convert metadata to JSON
            metadata_json = [json.dumps(m) for m in metadata]

            # Insert data into Milvus
            print("Metadata: ", metadata_json)
            insert_result = collection.insert([embeddings, metadata_json])
            collection.flush()
            
            # # Store the mapping from Milvus ID to text index
            # for milvus_id, text_index in zip(insert_result.primary_keys, range(num_embeddings)):
            #     milvus_id_to_text_index[milvus_id] = text_index
            
        collection.load()
        st.sidebar.success("Data inserted into Milvus successfully!")
    except Exception as e:
        st.error(f"Failed to insert data into Milvus: {e}")
        print(f"Failed to insert data into Milvus: {e}")

# Define retrieval functions using Milvus
def retrieve_documents(query_embedding, top_k=10):
    if not collection:
        print("Milvus collection is not initialized.")
        return []

    # Search for similar embeddings in Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(data=[query_embedding], anns_field="embeddings", param=search_params, limit=top_k, expr=None, output_fields=["metadata"])

    # metadataS = []
    # for result in results:
    #     indices.append([hit.id for hit in result])

    # Print debugging information
    # print(f"Query Embedding: {query_embedding}")
    # print(f"Meta Data: {metadataS}")

    return results

def rerank_documents(query, results):
    if not results:
        print("No results returned from retrieve_documents.")
        return []

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    query_pairs = [(query, json.loads(hit.entity.get("metadata")).get("chunk", "")) for result in results for hit in result]
    scores = cross_encoder.predict(query_pairs)
    ranked_hits = sorted(zip(query_pairs, scores), key=lambda x: x[1], reverse=True)
    return ranked_hits

# Function to create context from metadata
def create_context_from_metadata(results):
    context_chunks = []
    hierarchical_levels = {}
    
    for result in results:
        for hit in result:
            metadata_str = hit.entity.get("metadata")
            print("Metadata: ", metadata_json)
            if isinstance(metadata_str, dict):  # Check if metadata is already a dictionary
                metadata = metadata_str
            else:
                metadata = json.loads(metadata_str)  # Convert from JSON string to dictionary
            chunk = metadata.get("chunk", "")
            level = metadata.get("hierarchical_level", 0)
            if level not in hierarchical_levels:
                hierarchical_levels[level] = []
            hierarchical_levels[level].append(chunk)
    
    # Combine hierarchical chunks starting from the highest level
    for level in sorted(hierarchical_levels.keys()):
        context_chunks.extend(hierarchical_levels[level])
    
    context = " ".join(context_chunks)
    return context

# Find answer using GPT
def find_answer_gpt(question, context):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Based on the following context, please answer the question:\n\nContext: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=150,
            temperature=TEMPERATURE
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error during question answering: {e}")
        return "Error occurred."

# Streamlit interface for querying
query = st.text_input("Enter your query:", key="query_input")
if query and pdf_files:
    query_embedding = embed_chunks([query])[0]
    results = retrieve_documents(query_embedding)

    if results:
        with st.spinner("Assistant is typing..."):
            ranked_hits = rerank_documents(query, results)

            # Create context from metadata
            context = create_context_from_metadata(results)
        
            answer = find_answer_gpt(query, context)
            
            st.write(f"Question: {query}")
            st.write(f"Answer: {answer}")

    # if len(indices) > 0 and len(indices[0]) > 0:
    #     ranked_hits = rerank_documents(query, indices)
    #     if len(ranked_hits) > 0:
    #         context = " ".join([texts[idx] for idx, score in ranked_hits])
    #         answer = find_answer_gpt(query, context)
    #         st.write(f"Question: {query}")
    #         st.write(f"Answer: {answer}")
    #     else:
    #         st.write("No valid documents found for the query.")
    # else:
    #     st.write("No documents found for the query.")