# utils.py
import streamlit as st
import PyPDF2
import io
import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict

# Define constants here
KEYWORDS = {
    "Messaging Standards": ["MT format", "MX format", "ISO 20022", "FIN"],
    # ... other keywords
}

@st.cache_resource
def check_nltk_data():
    """Checks for NLTK data and downloads if missing. Returns True if successful."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        return True
    except LookupError:
        st.info("Downloading necessary NLTK data (punkt, stopwords)...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            st.success("NLTK data downloaded successfully.")
            return True
        except Exception as e:
            st.error(f"Failed to download NLTK data: {e}")
            return False

def load_css(file_name: str):
    """Loads an external CSS file."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    # ... (function is fine, keep as is)
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def extract_text_from_docs(uploaded_files: List) -> Dict:
    """Processes a list of uploaded files and returns a dictionary of their content."""
    docs = {}
    for file in uploaded_files:
        content = file.getvalue()
        raw_text = ""
        if file.type == "application/pdf":
            raw_text = extract_text_from_pdf(content)
        elif file.type == "text/plain":
            raw_text = content.decode("utf-8", errors='ignore')
        
        if raw_text:
            docs[file.name] = {
                "raw": raw_text,
                "processed": preprocess_text(raw_text),
                "sentences": get_sentences(raw_text)
            }
    return docs

def preprocess_text(text: str) -> str:
    # ... (function is fine, keep as is)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in tokens if word.isalnum() and word not in stop_words])

def get_sentences(text: str) -> List[str]:
    # ... (function is fine, keep as is)
    return [s.strip() for s in sent_tokenize(text) if s.strip()]