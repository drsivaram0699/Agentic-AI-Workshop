import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
import os

# --- Set page config ---
st.set_page_config(page_title="English to French Translator", layout="centered")

# --- Title ---
st.title("🌍 English to French Translator (Gemini + LangChain)")

# --- API Key Setup ---
st.markdown("🔐 AIzaSyDvEYJGn8911zs1J9EghdnmAMcfBjKKiz0")
api_key = st.text_input("Gemini API Key", type="password")

# --- Input ---
english_text = st.text_area("✏️ Enter English sentence to translate", height=150)

# --- Translate Button ---
if st.button("Translate"):
    if not api_key:
        st.error("❌ Please enter your Gemini API key.")
    elif not english_text.strip():
        st.error("❌ Please enter some text to translate.")
    else:
        try:
            # Set the environment variable for Gemini
            os.environ["GOOGLE_API_KEY"] = api_key

            # Initialize Gemini LLM
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

            # Prompt setup
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that translates English to French."),
                ("user", "Translate this sentence: {english_input}")
            ])

            # Create chain
            chain: Runnable = prompt | llm

            # Invoke chain
            result = chain.invoke({"english_input": english_text})

            # Extract response text
            translated_text = result.content.strip()

            # Show result
            st.success("✅ Translation completed!")
            st.markdown("### 🇫🇷 Translated French Sentence:")
            st.info(translated_text)

        except Exception as e:
            st.error(f"🚫 An error occurred: {str(e)}")
