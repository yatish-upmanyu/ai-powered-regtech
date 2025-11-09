import streamlit as st
import pandas as pd
import re
import io
from pathlib import Path

# --- Core ML/NLP Libraries ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# --- RAG and GenAI Libraries ---
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Document Processing ---
import PyPDF2

# --- UI/UX Libraries ---
from streamlit_lottie import st_lottie
import json


# =================================================================================================
# 1. PAGE CONFIG & INITIAL SETUP 
# =================================================================================================

st.set_page_config(
    page_title="Intelligent Compliance Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- App State Initialization ---
# This is crucial for a robust, multi-step app.
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "nltk_downloaded" not in st.session_state:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        st.session_state.nltk_downloaded = True
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        st.session_state.nltk_downloaded = False

# =================================================================================================
# 2. STYLING & ASSETS
# =================================================================================================

# --- Custom CSS for a modern look ---
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #F0F2F6;
    }
    /* Main title styling */
    .main-title {
        font-size: 2.8em;
        font-weight: bold;
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
        padding-top: 20px;
    }
    /* Subheader styling */
    .st-subheader {
        color: #4A5568; /* Gray */
    }
    /* Metric styling for dashboard */
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stMetric > label {
        font-weight: bold;
        color: #2D3748; /* Darker Gray */
    }
    .stMetric > div > div {
        font-size: 2.2em !important;
        color: #1E3A8A !important;
    }
    /* Styling for chat messages */
    .stChatMessage {
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# =================================================================================================
# 3. CORE LOGIC: ANALYSIS ENGINE (MODULAR & LEAN)
# =================================================================================================

class AnalysisEngine:
    """Encapsulates all backend document processing and analysis logic."""

    def __init__(self, uploaded_files):
        self.uploaded_files = uploaded_files
        self.docs = {}
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.KEYWORDS = {
            "Messaging Standards": ["MT format", "MX format", "ISO 20022", "FIN", "InterAct", "FileAct"],
            "Security Protocols": ["encryption", "authentication", "HSM", "PKI", "CSP", "CSCF"],
            "Risk & Compliance": ["risk assessment", "fraud detection", "AML", "CTF", "sanctions", "KYC"],
        }

    def _extract_text(self, file):
        """Extracts text from PDF or TXT files."""
        text = ""
        try:
            if file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            elif file.type == "text/plain":
                text = file.read().decode("utf-8")
        except Exception as e:
            st.warning(f"Could not read {file.name}: {e}")
        return text

    def _preprocess_text(self, text):
        """Cleans and tokenizes text for analysis."""
        if not text: return ""
        text = text.lower()
        text = re.sub(r'\d+', '', text) # Remove numbers
        text = re.sub(r'[\W_]+', ' ', text) # Remove punctuation and underscores
        tokens = [word for word in text.split() if word not in self.stopwords and len(word) > 2]
        return " ".join(tokens)

    def run_full_analysis(self, sim_threshold, diff_threshold):
        """Orchestrates the entire analysis pipeline."""
        # 1. Process documents
        for file in self.uploaded_files:
            raw_text = self._extract_text(file)
            sentences = nltk.sent_tokenize(raw_text)
            self.docs[file.name] = {
                "raw": raw_text,
                "processed": self._preprocess_text(raw_text),
                "sentences": [s.strip() for s in sentences if s.strip()]
            }
        
        doc_names = list(self.docs.keys())
        doc1 = self.docs[doc_names[0]]
        doc2 = self.docs[doc_names[1]]
        
        # 2. Run comparisons
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([doc1['processed'], doc2['processed']])
        
        overall_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        similar_pairs, diff1, diff2 = self._compare_sentences(
            doc1['sentences'], doc2['sentences'], sim_threshold, diff_threshold
        )
        
        # 3. Keyword and Gap analysis
        keywords_found = {name: self._find_keywords(data['raw']) for name, data in self.docs.items()}
        gap_report = self._generate_gap_report(keywords_found)
        
        return {
            "doc_names": doc_names,
            "overall_similarity": overall_similarity,
            "similar_pairs": similar_pairs,
            "differences_doc1": diff1,
            "differences_doc2": diff2,
            "keywords": keywords_found,
            "gaps": gap_report,
            "doc_data": self.docs
        }

    def _compare_sentences(self, sentences1, sentences2, sim_thresh, diff_thresh):
        """Finds similar and different sentences between two documents."""
        if not sentences1 or not sentences2:
            return [], sentences1, sentences2
            
        vectorizer = TfidfVectorizer().fit(sentences1 + sentences2)
        s1_vecs = vectorizer.transform(sentences1)
        s2_vecs = vectorizer.transform(sentences2)
        
        sim_matrix = cosine_similarity(s1_vecs, s2_vecs)
        
        similar_pairs = []
        for i, row in enumerate(sim_matrix):
            best_match_idx = row.argmax()
            if row[best_match_idx] > sim_thresh:
                similar_pairs.append((sentences1[i], sentences2[best_match_idx], row[best_match_idx]))
        
        diff1 = [s1 for i, s1 in enumerate(sentences1) if sim_matrix[i].max() < diff_thresh]
        sim_matrix_t = sim_matrix.T
        diff2 = [s2 for j, s2 in enumerate(sentences2) if sim_matrix_t[j].max() < diff_thresh]
        
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs, diff1, diff2

    def _find_keywords(self, text):
        """Identifies predefined keywords in text."""
        found = {}
        text_lower = text.lower()
        for category, kws in self.KEYWORDS.items():
            found_kws = [kw for kw in kws if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower)]
            if found_kws:
                found[category] = found_kws
        return found
        
    def _generate_gap_report(self, keywords_found):
        """Compares keyword presence across documents to find gaps."""
        doc_names = list(keywords_found.keys())
        all_kws = set(kw for data in keywords_found.values() for cat_kws in data.values() for kw in cat_kws)
        
        gap_report = []
        for kw in all_kws:
            present_in = [name for name, data in keywords_found.items() if any(kw in cat_kws for cat_kws in data.values())]
            if 0 < len(present_in) < len(doc_names):
                missing_in = list(set(doc_names) - set(present_in))
                gap_report.append({
                    "Keyword": kw,
                    "Present In": ", ".join(present_in),
                    "Missing In": ", ".join(missing_in)
                })
        return pd.DataFrame(gap_report)

# =================================================================================================
# 4. RAG IMPLEMENTATION
# =================================================================================================

@st.cache_resource(show_spinner="Setting up RAG Pipeline...")
def setup_rag_pipeline(doc_data): # <--- REMOVE azure_config from parameters
    """Creates and caches the RAG pipeline (vector store and QA chain)."""
    # --- Access secrets INSIDE the function ---
    azure_config = st.secrets["azure"]
    # ----------------------------------------
    
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for doc_name, data in doc_data.items():
        chunks = text_splitter.create_documents([data['raw']], metadatas=[{"source": doc_name}])
        all_chunks.extend(chunks)

    # Configure Azure OpenAI
    openai.api_type = "azure"
    openai.api_key = azure_config['api_key'] # <--- Now uses the locally defined variable
    openai.api_base = azure_config['endpoint']
    openai.api_version = azure_config['api_version']
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_config['embedding_deployment'],
        chunk_size=1
    )
    
    llm = AzureChatOpenAI(
        deployment_name=azure_config['chat_deployment'],
        temperature=0.1,
        max_tokens=1000
    )
    
    # Create Vector Store
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    
    # Create QA Chain
    prompt_template = """You are an expert compliance analyst AI. Use the following pieces of context from the uploaded documents to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Provide a concise and direct answer, then list the sources you used.

    Context:
    {context}

    Question: {question}
    
    Answer:"""
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    return qa_chain

# =================================================================================================
# 5. UI RENDERING FUNCTIONS
# =================================================================================================


def render_uploader():
    """Displays the initial file uploader interface (without animation)."""
    st.markdown('<p class="main-title">Intelligent Compliance Analyzer</p>', unsafe_allow_html=True)
    st.markdown("<p class='st-subheader' style='text-align: center;'>Compare SWIFT documents, identify gaps, and chat with your data using AI.</p>", unsafe_allow_html=True)
    
    # --- Simple Icon instead of Animation ---
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <span style="font-size: 5em; color: #1E3A8A;">📄</span>
        <span style="font-size: 5em; color: #3B82F6;">📑</span>
    </div>
    """, unsafe_allow_html=True)
    # --- End of Icon ---

    uploaded_files = st.file_uploader(
        "Upload 2 PDF or TXT documents for comparison",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    return uploaded_files

def render_dashboard(results):
    """Renders the main dashboard with key metrics."""
    st.header("Executive Dashboard")
    doc1, doc2 = results['doc_names']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Overall Similarity", value=f"{results['overall_similarity']:.0%}")
    with col2:
        st.metric(label=f"Similar Content Areas", value=len(results['similar_pairs']))
    with col3:
        st.metric(label=f"Unique Items in {Path(doc1).stem}", value=len(results['differences_doc1']))
    with col4:
        st.metric(label=f"Potential Gaps Found", value=len(results['gaps']))
        
def render_analysis_tabs(results):
    """Renders the detailed analysis in a tabbed interface."""
    doc1_name, doc2_name = results['doc_names']
    
    tab1, tab2, tab3 = st.tabs(["📊 Side-by-Side Comparison", "🔑 Keyword & Gap Analysis", "💬 Chat with Your Docs (RAG)"])
    
    with tab1:
        st.subheader("Document Similarity Breakdown")
        st.info(f"Comparing **{doc1_name}** and **{doc2_name}**.")
        
        st.markdown("#### Top Similarities")
        if results['similar_pairs']:
            df_similar = pd.DataFrame(results['similar_pairs'], columns=[doc1_name, doc2_name, 'Score'])
            df_similar['Score'] = df_similar['Score'].map('{:.2%}'.format)
            st.dataframe(df_similar.head(), use_container_width=True, hide_index=True)
        else:
            st.warning("No highly similar sentences found based on the current threshold.")
            
        st.markdown(f"#### Content Unique to __{doc1_name}__")
        if results['differences_doc1']:
            for sent in results['differences_doc1'][:5]:
                st.markdown(f"- {sent}")
        else:
            st.success("No significant unique content found.")
            
        st.markdown(f"#### Content Unique to __{doc2_name}__")
        if results['differences_doc2']:
            for sent in results['differences_doc2'][:5]:
                st.markdown(f"- {sent}")
        else:
            st.success("No significant unique content found.")

    with tab2:
        st.subheader("Keyword Presence and Gap Analysis")
        st.markdown("This section highlights the presence of key compliance terms and identifies where they might be missing.")
        
        st.markdown("#### Potential Gaps")
        if not results['gaps'].empty:
            st.dataframe(results['gaps'], use_container_width=True, hide_index=True)
        else:
            st.success("No keyword-based gaps were identified between the documents.")
        
        st.markdown("#### Keyword Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Keywords in {doc1_name}**")
            st.json(results['keywords'][doc1_name], expanded=False)
        with col2:
            st.write(f"**Keywords in {doc2_name}**")
            st.json(results['keywords'][doc2_name], expanded=False)

    with tab3:
        render_chat_ui()


def render_chat_ui():
    """Renders the RAG chat interface."""
    st.subheader("Chat with Your Documents")
    st.info("Ask a question about the content of your uploaded documents. The AI will answer based on the provided text and cite its sources.")

    if st.session_state.rag_pipeline is None:
        st.warning("RAG pipeline not initialized. Please ensure documents are analyzed.")
        return

    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for source in msg["sources"]:
                        st.info(f"**From: {source.metadata['source']}**\n\n> {source.page_content}")

    # Accept user input
    if prompt := st.chat_input("Ask about compliance, regulations, or specific terms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_pipeline({"query": prompt})
                    answer = response['result']
                    sources = response['source_documents']
                    
                    st.markdown(answer)
                    with st.expander("View Sources"):
                        for source in sources:
                             st.info(f"**From: {source.metadata['source']}**\n\n> {source.page_content}")
                    
                    # Store response with sources for redisplay
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"An error occurred while querying the AI: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I ran into an error: {e}"
                    })


# =================================================================================================
# 6. MAIN APPLICATION FLOW
# =================================================================================================

# =================================================================================================
# 6. MAIN APPLICATION FLOW
# =================================================================================================

def main():
    """Main function to run the Streamlit app."""

    # --- Sidebar for controls ---
    with st.sidebar:
        # --- Hardcoded SVG Logo ---
        logo_svg = """
        <svg width="100%" height="80" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#1E3A8A;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#3B82F6;stop-opacity:1" />
                </linearGradient>
            </defs>
            <rect width="100%" height="100%" rx="10" ry="10" fill="url(#grad1)" />
            <text x="50%" y="50%" font-family="'Segoe UI', 'Roboto', sans-serif" font-size="22" font-weight="bold" fill="white" text-anchor="middle" dominant-baseline="middle">
                Regulatory Platform
            </text>
        </svg>
        """
        st.markdown(logo_svg, unsafe_allow_html=True)
        # --- End of SVG Logo ---

        st.header("⚙️ Analysis Controls")
        
        sim_threshold = st.slider(
            "Similarity Threshold", 0.5, 1.0, 0.8, 0.05,
            help="How similar sentences must be to be considered a 'match'. Higher is stricter."
        )
        diff_threshold = st.slider(
            "Difference Threshold", 0.0, 0.5, 0.2, 0.05,
            help="How different a sentence must be to be considered 'unique'. Lower is stricter."
        )
        st.markdown("---")
        st.info("Built with ❤️ using Streamlit, LangChain, and Azure OpenAI.")

    # --- Main content area ---
    if st.session_state.analysis_results is None:
        uploaded_files = render_uploader()
        
        if uploaded_files and len(uploaded_files) == 2:
            if st.button("🚀 Analyze Documents", use_container_width=True, type="primary"):
                with st.spinner("Analyzing documents, please wait..."):
                    try:
                        # 1. Run core analysis
                        engine = AnalysisEngine(uploaded_files)
                        results = engine.run_full_analysis(sim_threshold, diff_threshold)
                        st.session_state.analysis_results = results
                        
                        # 2. Setup RAG pipeline
                        st.session_state.rag_pipeline = setup_rag_pipeline(results['doc_data'])
                        
                        # 3. Clear chat history for new analysis
                        st.session_state.messages = []
                        
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                        st.session_state.analysis_results = None # Reset on failure

        elif uploaded_files:
            st.warning("Please upload exactly two documents for comparison.")
    
    else:
        # --- Display results if analysis is complete ---
        render_dashboard(st.session_state.analysis_results)
        st.markdown("---")
        render_analysis_tabs(st.session_state.analysis_results)
        
        if st.sidebar.button("Start New Analysis"):
            # Clear all state to start over
            st.session_state.analysis_results = None
            st.session_state.rag_pipeline = None
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    if st.session_state.nltk_downloaded:
        main()
    else:
        st.error("Application cannot start because required NLTK data is missing. Please check your internet connection and restart.")