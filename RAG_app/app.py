import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

# --- LOAD ENV ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Set it in environment variables.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 Chat with Your Document")

# --- INPUT ---
query = st.text_input("💬 Ask your question")

uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])
url = st.text_input("🌐 Or paste document URL")

# --- CACHE EMBEDDINGS ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings()

# --- CACHE MODEL ---
@st.cache_resource
def load_model():
    return init_chat_model(
        "google_genai:gemini-3-flash-preview",
        api_key=api_key
    )

embeddings = load_embeddings()
model = load_model()

# --- INITIALIZE SESSION STATE ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None

# --- PROCESS DOCUMENT ONLY ONCE ---
def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        loader = PyPDFLoader(tmp.name)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    vectordb = Chroma.from_documents(docs, embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 2})

def process_url(web_url):
    loader = WebBaseLoader(web_url)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    vectordb = Chroma.from_documents(docs, embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 2})

# --- BUTTON ---
if st.button("Get Answer"):

    # --- VALIDATION ---
    if not query:
        st.warning("Enter a question")
        st.stop()

    if uploaded_file and url:
        st.error("Choose either file OR URL, not both")
        st.stop()

    if not uploaded_file and not url:
        st.warning("Upload file or provide URL")
        st.stop()

    with st.spinner("Processing..."):

        # --- GET OR PROCESS DOCUMENT ---
        if uploaded_file:
            doc_id = uploaded_file.name
            if st.session_state.current_doc_id != doc_id:
                st.session_state.retriever = process_pdf(uploaded_file.read())
                st.session_state.current_doc_id = doc_id
        else:
            doc_id = url
            if st.session_state.current_doc_id != doc_id:
                st.session_state.retriever = process_url(url)
                st.session_state.current_doc_id = doc_id

        retriever = st.session_state.retriever

        # --- RETRIEVE ---
        retrieved_docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # --- PROMPT ---
        prompt = f"""
        Answer ONLY from the given context.
        If answer not found, say "Not found in document".

        Context:
        {context}

        Question:
        {query}
        """

        # --- LLM CALL ---
        response = model.invoke(prompt)
        
        # Extract text content properly
        if isinstance(response.content, list):
            answer_text = response.content[0].get('text', str(response.content)) if response.content else "No answer found"
        else:
            answer_text = str(response.content)

        # --- OUTPUT ---
        st.write(answer_text)