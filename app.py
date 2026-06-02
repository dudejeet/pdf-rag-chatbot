import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")

st.title("AI-Powered PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_path = "uploaded_file.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    question = st.text_input("Ask a question about the PDF:")

    if question:
        result = qa_chain.invoke({"query": question})

        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Source Chunks")
        for doc in result["source_documents"]:
            st.write(doc.page_content[:500])
            st.write("---")
