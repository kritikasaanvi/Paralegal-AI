import os

import pdfplumber
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import Together

# Streamlit page configuration
st.set_page_config(page_title="AI PARALEGAL", layout="wide")
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("BHARAT LEX.jpg")

# Streamlit style settings
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, black, #ff6600);
        color: white;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    div.stButton > button:first-child {
        background-color: #ff6600;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff4500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Reset function
def reset_conversation():
    st.session_state.clear()
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", return_messages=True
    )
    st.experimental_rerun()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", return_messages=True
    )

# Initialize embeddings and retriever
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"},
)

db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define LLM prompt template
prompt_template = """<s>[INST]This is a legal chat bot specializing in Indian Penal Code queries...
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

# Initialize Together AI
os.environ["TOGETHER_AI"] = "3b01ba7c029199b51dfa32baa6aff8c3d261a60c4552c05dac17b95b2c7bf964"
TOGETHER_AI_API = os.environ["TOGETHER_AI"]
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.5, max_tokens=1024, together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, memory=st.session_state.memory, retriever=db_retriever, combine_docs_chain_kwargs={"prompt": prompt}
)

# PDF Summarization Function
def summarize_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()

        # Chunk the text for better LLM processing
        max_chunk_size = 1000
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        total_chunks = len(chunks)
        summaries = []

        # Progress display
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Concise summarization prompt
        summarization_prompt = """<s>[INST]
Summarize the following legal text into 2-3 concise sentences:
{content}
</s>[INST]"""

        for i, chunk in enumerate(chunks):
            status_text.text(f"Processing chunk {i + 1}/{total_chunks}...")
            prompt = summarization_prompt.format(content=chunk)
            summary = llm.invoke(prompt)
            summaries.append(summary.strip())
            progress_bar.progress((i + 1) / total_chunks)

        status_text.text("PDF summarization complete!")
        progress_bar.empty()

        return "\n\n".join(summaries)  # Join with double newlines for readability
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return None

# Chat display function
def display_chat_messages(messages):
    for message in messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

# Process user input
def process_user_input(input_text):
    with st.chat_message("user"):
        st.write(input_text)

    st.session_state.messages.append({"role": "user", "content": input_text})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke(input=input_text)
            st.write(result["answer"])
    
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# Scrape legal news function
def scrape_legal_news():
    news_data = []
    try:
        url = "https://indianexpress.com/section/india/legal/"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = soup.find_all('div', class_='articles')
            for item in news_items[:5]:  # Top 5 news items
                headline_elem = item.find('h2', class_='title')
                summary_elem = item.find('p', class_='description')
                link_elem = item.find('a', class_='story')
                if headline_elem and summary_elem and link_elem:
                    headline = headline_elem.text.strip()
                    summary = summary_elem.text.strip()
                    link = link_elem['href']
                    news_data.append((headline, summary, link))
    except Exception as e:
        st.error(f"An error occurred while scraping legal news: {e}")

    return news_data

# Check for reset flag
if st.button("Reset All Chat 🗑️"):
    reset_conversation()

# Main Chat Display Section
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# Input Section
input_prompt = st.text_input("Say something", key="user_input")

if input_prompt:
    process_user_input(input_prompt)

# PDF Upload Section
st.subheader("Upload Case Law PDF")
pdf_file = st.file_uploader("Upload your case law (PDF format)", type=["pdf"])
if pdf_file:
    summary = summarize_pdf(pdf_file)
    if summary:
        st.write("### Case Law Summary")
        st.write(summary)

# Legal News Section
st.subheader("Legal News Updates")
st.write("Stay informed with the latest legal news!")

legal_news = scrape_legal_news()

if legal_news:
    for headline, summary, link in legal_news:
        st.subheader(headline)
        st.write(summary)
        st.write(f"[Read more]({link})")
        st.write("---")
else:
    st.write("No legal news updates available at the moment.")
