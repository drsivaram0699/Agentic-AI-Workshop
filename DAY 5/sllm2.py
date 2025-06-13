import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# --- 1. Set your Gemini API Key ---
GEMINI_API_KEY = "AIzaSyDvEYJGn8911zs1J9EghdnmAMcfBjKKiz0"  # Replace with your Gemini API key

# --- 2. Streamlit App Setup ---
st.set_page_config(page_title="📄 Smart PDF + General Q&A", layout="centered")
st.title("📘 PDF Question Answering + General Chat with RAG Status")

# --- 3. Session State Setup ---
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "rag_status" not in st.session_state:
    st.session_state.rag_status = "🔴 Awaiting PDF upload..."

# --- 4. File Upload ---
uploaded_file = st.file_uploader("📤 Upload a PDF file (optional)", type="pdf")

if uploaded_file:
    st.session_state.rag_status = "🟠 Processing PDF..."
    st.info(st.session_state.rag_status)
    try:
        pdf = PdfReader(uploaded_file)
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(full_text)
        st.session_state.doc_chunks = chunks
        st.session_state.rag_status = "🟢 PDF ready! Ask your question."
    except Exception as e:
        st.session_state.rag_status = f"🔴 Error: {str(e)}"
        st.error(st.session_state.rag_status)
else:
    st.session_state.rag_status = "🟡 No PDF uploaded. General chat mode is enabled."

# --- 5. Display RAG Status ---
st.markdown(f"### ✅ System Status: {st.session_state.rag_status}")

# --- 6. Question Input ---
user_question = st.text_input("❓ Ask a question (about PDF or anything):")

# --- 7. Answering Logic ---
if st.button("Get Answer") and user_question.strip():
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=GEMINI_API_KEY
        )

        # PDF context if available
        pdf_context = "\n\n".join(st.session_state.doc_chunks[:5]) if st.session_state.doc_chunks else ""

        # Prompt dynamically adjusted
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a smart assistant. If PDF context is provided, use it. Otherwise, answer based on your general knowledge."),
            ("user", "PDF Context:\n{context}\n\nQuestion: {question}")
        ])

        chain: Runnable = prompt | llm

        result = chain.invoke({
            "context": pdf_context,
            "question": user_question
        })

        st.success("✅ Answer:")
        st.write(result.content.strip())

    except Exception as e:
        st.error(f"❌ Error while answering: {str(e)}")
