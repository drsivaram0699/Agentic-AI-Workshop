import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

# 1. Set your Google Gemini API key (hardcoded)
GEMINI_API_KEY = "AIzaSyDvEYJGn8911zs1J9EghdnmAMcfBjKKiz0"  # 🔐 Replace with your actual Gemini API key

# 2. Configure Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY
)

# 3. Define DuckDuckGo Search tool
search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    func=search.run,
    description="Useful for answering questions about current events or real-time topics from the web."
)

# 4. Initialize agent with Gemini + DuckDuckGo tool
agent = initialize_agent(
    tools=[duckduckgo_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False  # Prevents logs in terminal
)

# 5. Streamlit UI setup
st.set_page_config(page_title="🌐 Real-Time Q&A with Gemini + DuckDuckGo", layout="centered")
st.title("🧠 Ask Anything: Gemini + 🦆 DuckDuckGo")

st.markdown("Type your question below. I’ll try my best to answer using real-time info 🌍")

# 6. Input
user_query = st.text_input("💬 Enter your question here:", placeholder="e.g. What's the latest on AI regulations?")
ask_button = st.button("🔍 Get Answer")

# 7. Run query
if ask_button and user_query.strip():
    with st.spinner("Thinking..."):
        try:
            response = agent.run(user_query)
            st.success("✅ Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Oops! Something went wrong: {str(e)}")
