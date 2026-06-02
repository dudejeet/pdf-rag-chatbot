import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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

    question = st.text_input("Ask a question about the PDF:")

    if question:
        relevant_docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = ChatPromptTemplate.from_template(
            """
            You are an AI assistant that answers questions using only the provided PDF context.

            Context:
            {context}

            Question:
            {question}

            Answer clearly and accurately based only on the context.
            If the answer is not found in the context, say:
            "I could not find that information in the PDF."
            """
        )

        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        st.subheader("Answer")
        st.write(response.content)

        st.subheader("Source Chunks")
        for doc in relevant_docs:
            st.write(doc.page_content[:500])
            st.write("---")
