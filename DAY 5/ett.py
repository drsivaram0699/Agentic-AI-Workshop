import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# --- SET YOUR GOOGLE GEMINI API KEY HERE ---
GEMINI_API_KEY = "AIzaSyDvEYJGn8911zs1J9EghdnmAMcfBjKKiz0"  # 🔐 Replace this with your Gemini API Key

# --- Streamlit UI Setup ---
st.set_page_config(page_title="English to Tamil Translator", layout="centered")
st.title("🌐 English to Tamil Translator using Gemini + LangChain")

# --- Input Section ---
user_input = st.text_area("✏️ Enter an English sentence to translate:", height=150)
translate_button = st.button("Translate to Tamil")

# --- Translation Process ---
if translate_button:
    if not user_input.strip():
        st.error("❌ Please enter a sentence to translate.")
    else:
        try:
            # Step 1: Define the LLM with API key
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                google_api_key=GEMINI_API_KEY
            )

            # Step 2: Define the translation prompt (English → Tamil)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that translates English to Tamil."),
                ("user", "Translate this sentence: {english_input}")
            ])

            # Step 3: Chain the prompt with the model
            chain: Runnable = prompt | llm

            # Step 4: Invoke the chain with user input
            result = chain.invoke({"english_input": user_input})

            # Step 5: Display the output
            st.success("✅ Translation completed!")
            st.markdown("### 🈷️ Translated Tamil Sentence:")
            st.info(result.content.strip())

        except Exception as e:
            st.error(f"🚫 Error during translation: {str(e)}")
