import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# --- SET YOUR GOOGLE GEMINI API KEY HERE ---
GEMINI_API_KEY = "AIzaSyDvEYJGn8911zs1J9EghdnmAMcfBjKKiz0"  # Replace with your actual Gemini API key

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🌐 English Translator", layout="centered")
st.title("🌍 English to Multiple Language Translator using Gemini + LangChain")

# --- Language Dropdown ---
languages = ["French", "Tamil", "Spanish", "Hindi", "German", "Chinese"]
target_language = st.selectbox("🌏 Choose the language to translate into:", languages)

# --- Input Section ---
user_input = st.text_area("✏️ Enter an English sentence to translate:", height=150)
translate_button = st.button("🔁 Translate")

# --- Translation Logic ---
if translate_button:
    if not user_input.strip():
        st.error("❌ Please enter a sentence to translate.")
    else:
        try:
            # 1. Define Gemini LLM with API key
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                google_api_key=GEMINI_API_KEY
            )

            # 2. Define Prompt Template for multi-language translation
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that translates English into the language specified."),
                ("user", "Translate this English sentence to {target_language}:\n\n{english_input}")
            ])

            # 3. Create the Runnable chain
            chain: Runnable = prompt | llm

            # 4. Run the chain with input values
            result = chain.invoke({
                "english_input": user_input,
                "target_language": target_language
            })

            # 5. Show Result
            st.success("✅ Translation completed!")
            st.markdown(f"### 🈯 Translated Sentence in {target_language}:")
            st.info(result.content.strip())

        except Exception as e:
            st.error(f"🚫 Error during translation: {str(e)}")
