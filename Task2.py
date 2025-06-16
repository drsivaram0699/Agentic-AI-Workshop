# app.py  –  AI-Powered PDF Plagiarism Checker (English Only)
# -----------------------------------------------------------
# Prerequisites:
#   pip install streamlit google-generativeai langdetect pypdf
# -----------------------------------------------------------

import streamlit as st
import google.generativeai as genai
from langdetect import detect, LangDetectException
from pypdf import PdfReader
import re
import io

# ---------------------------
# Gemini API Configuration
# ---------------------------
GOOGLE_API_KEY = "AIzaSyAalfP3WDEuVVIlKgTHB6odfKLm1n5sU5A"   # <-- replace / secure in production
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel("gemini-2.0-flash")
MAX_CHARS = 20_000            # safety limit for one request

# ---------------------------
# Helper Functions
# ---------------------------
def is_english(text: str) -> bool:
    """Detect whether *text* is English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def extract_text_from_pdf(pdf_file) -> str:
    """Read PDF (uploaded file-like object) and return concatenated text."""
    reader = PdfReader(pdf_file)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text).strip()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="📄 Gemini Flash PDF Plagiarism Checker", layout="centered")
st.title("📄 AI-Powered Plagiarism Checker (English, PDF Upload)")

with st.form("plagiarism_form"):
    name   = st.text_input("Full Name *")
    email  = st.text_input("Email Address *")
    domain = st.selectbox(
        "Academic / Research Domain *",
        ["Arts", "Engineering", "Pharmacy", "Allied", "Physio", "Nursing"]
    )
    pdf_file = st.file_uploader(
        "Upload your manuscript (PDF, English only) *",
        type=["pdf"],
        accept_multiple_files=False,
    )
    submitted = st.form_submit_button("Check for Plagiarism")

# ---------------------------
# Validation & Processing
# ---------------------------
if submitted:
    errors = []

    # — Basic field checks —
    if not name.strip():
        errors.append("Name is required.")
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        errors.append("A valid email address is required.")
    if pdf_file is None:
        errors.append("Please upload a PDF file.")
    
    # — Extract and validate PDF text —
    pdf_text = ""
    if pdf_file is not None:
        try:
            pdf_bytes = io.BytesIO(pdf_file.read())
            pdf_text  = extract_text_from_pdf(pdf_bytes)
            if not pdf_text:
                errors.append("Could not extract text from the PDF.")
            elif not is_english(pdf_text[:5000]):  # sample first 5 000 chars
                errors.append("Uploaded document appears not to be in English.")
            elif len(pdf_text) > MAX_CHARS:
                errors.append("Document is too large for a single check; please shorten it.")
        except Exception as e:
            errors.append(f"Error reading PDF: {e}")

    # — Display errors or run Gemini —
    if errors:
        for err in errors:
            st.error(err)
    else:
        with st.spinner("Running Gemini Flash plagiarism analysis…"):
            prompt = (
                "You are an academic plagiarism detector. Analyse the following English text, "
                "identify potential plagiarism, list similar sources with brief citations if possible, "
                "and provide an overall originality percentage (0-100). "
                "Keep your answer under 250 words.\n\n"
                f"--- BEGIN DOCUMENT ---\n{pdf_text}\n--- END DOCUMENT ---"
            )
            try:
                response = MODEL.generate_content(prompt)
                st.success("Plagiarism Check Completed ✅")
                st.markdown("### 🔍 Result")
                st.write(response.text)
            except Exception as e:
                st.error(f"Gemini API error: {e}")
