import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.set_page_config(page_title="AI Compliance Agent", layout="centered")
st.title("🛡️ AI Compliance Agent")
st.write("Upload a compliance document and ask questions about it!")

uploaded_file = st.file_uploader("📄 Upload a .txt file", type=["txt"])

if uploaded_file:
    with open("uploaded.txt", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split the text
    loader = TextLoader("uploaded.txt", encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # Embed and create FAISS vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embedding_model)
    retriever = vectorstore.as_retriever()

    # User query
    query = st.text_input("💬 Ask a question about the document:")

    if query:
        results = retriever.get_relevant_documents(query)
        if results:
            st.subheader("📌 Answer")
            st.write(results[0].page_content)
        else:
            st.warning("No relevant content found.")
else:
    st.info("Please upload a .txt file to get started.")
