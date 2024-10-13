import streamlit as st
import os
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import tempfile

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
# Load environment variables from .env file
load_dotenv()

# # Set Pinecone API key and Groq API key
# api_key = os.getenv("PINECONE_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")

api_key = st.secrets["PINECONE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize Pinecone client and create index if it doesn't exist
index_name = "langchain-pinecone-hybrid-search"
pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Embeddings and sparse matrix (BM25)
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
bm25_encoder = BM25Encoder().default()

# Streamlit app header
st.title("PDF Hybrid Search with Pinecone & Groq")

# PDF Upload
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Process the PDFs and retrieve content
if uploaded_files:
    dataset = []
    for uploaded_file in uploaded_files:
        # Use a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        # Load the PDF using PyPDFLoader
        pdf_loader = PyPDFLoader(temp_file_path)
        docs = pdf_loader.load()
        
        for doc in docs:
            dataset.append(doc.page_content)

    # Fit BM25 on the dataset
    bm25_encoder.fit(dataset)

    # Initialize hybrid search retriever
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
    )
    retriever.add_texts(dataset)

    # Initialize Groq-based QA model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    chain = load_qa_chain(llm, chain_type="stuff")

    # User Query
    query = st.text_input("Ask a question based on the uploaded PDFs:")

    if query:
        # Retrieve similar documents and pass to chain for answer generation
        similar_docs = retriever.invoke(query)
        answer = chain.invoke({"input_documents": similar_docs, "question": query})

        # Display the answer
        st.write("Answer:", answer['output_text'])

else:
    st.write("Please upload PDFs to start the search.")
