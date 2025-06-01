import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

st.set_page_config(page_title="AI Compliance Agent", layout="centered")
st.title("🛡️ AI Compliance Agent")
st.write("Upload compliance documents and ask questions about them!")

uploaded_file = st.file_uploader("📄 Upload a .txt file", type=["txt"])

if uploaded_file:
    # Save uploaded file
    with open("uploaded.txt", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split
    loader = TextLoader("uploaded.txt", encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # Embeddings and vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embedding_model)

    retriever = vectorstore.as_retriever()

    # User query
    query = st.text_input("💬 Ask a question:")

    if query:
        results = retriever.get_relevant_documents(query)
        if results:
            st.subheader("📌 Answer")
            st.write(results[0].page_content)
        else:
            st.warning("No relevant information found.")
else:
    st.info("Please upload a .txt file to get started.")
