import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# --- 1. Set your Gemini API Key ---
GEMINI_API_KEY = "AIzaSyDvEYJGn8911zs1J9EghdnmAMcfBjKKiz0"  # Replace with your Gemini API key

# --- 2. Streamlit App Setup ---
st.set_page_config(page_title="📄 PDF QA with RAG Status", layout="centered")
st.title("📘 PDF Question Answering System with RAG Indicator")

# --- 3. Session State Setup ---
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "rag_status" not in st.session_state:
    st.session_state.rag_status = "🔴 Awaiting PDF upload..."

# --- 4. File Upload ---
uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

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

# --- 5. Display RAG Status ---
st.markdown(f"### ✅ System Status: {st.session_state.rag_status}")

# --- 6. Question Answering Interface ---
if st.session_state.doc_chunks:
    user_question = st.text_input("❓ Enter your question about the PDF:")
    if st.button("Get Answer") and user_question.strip():
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # ✅ corrected model name
                temperature=0.3,
                google_api_key=GEMINI_API_KEY
            )

            context = "\n\n".join(st.session_state.doc_chunks[:5])  # basic context for now

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant that answers questions based on PDF content."),
                ("user", "Context:\n{context}\n\nQuestion: {question}")
            ])

            chain: Runnable = prompt | llm

            result = chain.invoke({
                "context": context,
                "question": user_question
            })

            st.success("✅ Answer:")
            st.write(result.content.strip())

        except Exception as e:
            st.error(f"❌ Error while answering: {str(e)}")
