import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Ask My PDF Bot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Ask My PDF Bot")
st.write("Upload a PDF and ask questions about its content.")


# -------------------------------
# Hugging Face API Key
# -------------------------------

hf_token = st.sidebar.text_input(
    "Enter Hugging Face API Token",
    type="password"
)

if not hf_token:
    st.sidebar.info(
        "Enter your Hugging Face API token to use the chatbot."
    )
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name


    # -------------------------------
    # Load PDF
    # -------------------------------

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()


    # -------------------------------
    # Split PDF into chunks
    # -------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)


    # -------------------------------
    # Create Embeddings
    # -------------------------------

    with st.spinner("Creating document embeddings..."):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


        # -------------------------------
        # Create FAISS Vector Database
        # -------------------------------

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )


    st.success("PDF processed successfully! 🎉")


    # -------------------------------
    # LLM
    # -------------------------------

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={
            "temperature": 0.2,
            "max_length": 512
        }
    )


    # -------------------------------
    # Prompt
    # -------------------------------

    prompt_template = """
    Use the following context to answer the question.

    If the answer cannot be found in the context,
    say "I could not find the answer in the PDF."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


    # -------------------------------
    # RAG Chain
    # -------------------------------

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}
        ),
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=True
    )


    # -------------------------------
    # Ask Question
    # -------------------------------

    question = st.text_input(
        "Ask a question about your PDF:"
    )


    if question:

        with st.spinner("Searching the PDF..."):

            result = qa_chain.invoke({
                "query": question
            })


        st.subheader("🤖 Answer")

        st.write(result["result"])


        # -------------------------------
        # Sources
        # -------------------------------

        st.subheader("📄 Sources")

        source_documents = result.get(
            "source_documents",
            []
        )

        for doc in source_documents:

            page_number = doc.metadata.get(
                "page",
                "Unknown"
            )

            st.write(
                f"Page: {page_number + 1}"
                if isinstance(page_number, int)
                else f"Page: {page_number}"
            )