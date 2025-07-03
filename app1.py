# app1.py

import streamlit as st
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")

# Setup tools
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

tools = {
    "Wikipedia": wiki,
    "Arxiv": arxiv
}

# Set up embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Streamlit App
st.set_page_config(page_title="LLM Tool App", layout="centered")
st.title("📚 Research Assistant App")
st.markdown("Query from Arxiv or Wikipedia, and explore documents using embeddings.")

# Input query
query = st.text_input("🔍 Enter your research question:")
tool_option = st.radio("Select source:", list(tools.keys()))
search_btn = st.button("Search")

# Tool run
if search_btn and query:
    with st.spinner("Searching..."):
        result = tools[tool_option].run(query)
        st.success("Results fetched successfully!")
        st.markdown("### 📄 Result:")
        st.write(result)

# Extra: Load and embed web document (static example for now)
st.markdown("---")
st.markdown("### 🧪 Test Embedding Example")

if st.button("Load & Embed Example Web Page"):
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Natural_language_processing")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    with st.spinner("Embedding & indexing..."):
        result = retriever.get_relevant_documents("What is NLP?")
        st.success("Embedding complete. Retrieved context:")

        for i, doc in enumerate(result):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
