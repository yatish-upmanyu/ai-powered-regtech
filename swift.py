import streamlit as st
import pandas as pd
import PyPDF2
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(
    page_title="SWIFT Compliance Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Coral Orange Theme ---
primary_color = "#FF7F50"  # Coral Orange
secondary_color = "#FFA07A"  # Light Coral
background_color = "#F5F5F5"  # Light Gray
text_color = "#333333"      # Dark Gray
accent_color = "#8B4513"    # Saddle Brown (for contrast)

st.markdown(
    f"""
    <style>
    body {{
        background-color: {background_color};
        color: {text_color};
    }}
    .st-header {{
        background-color: rgba(255, 127, 80, 0.1); /* Subtle coral header */
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }}
    .st-subheader {{
        color: {primary_color};
        margin-top: 15px;
    }}
    .st-info {{
        background-color: #FFE4B5; /* Light Sandy Brown */
        border-left: 5px solid {secondary_color};
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 3px;
    }}
    .st-warning {{
        background-color: #FFDAB9; /* Peach Puff */
        border-left: 5px solid #FF8C00; /* Dark Orange */
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 3px;
    }}
    .st-success {{
        background-color: #F0FFF0; /* Honeydew */
        border-left: 5px solid #3CB371; /* Medium Sea Green */
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 3px;
    }}
    .st-metric-label {{
        color: {accent_color};
    }}
    .st-metric-value {{
        color: {primary_color};
        font-size: 1.6em;
        font-weight: bold;
    }}
    .st-expander {{
        border: 1px solid {secondary_color};
        border-radius: 5px;
        margin-bottom: 10px;
    }}
    .st-expander-header {{
        color: {primary_color};
        font-weight: bold;
    }}
    .st-markdown h1 {{
        color: {primary_color};
    }}
    .st-markdown h2 {{
        color: {primary_color};
    }}
    .st-markdown h3 {{
        color: {primary_color};
    }}
    .st-markdown h4 {{
        color: {primary_color};
    }}
    .st-markdown h5 {{
        color: {primary_color};
    }}
    .st-markdown h6 {{
        color: {primary_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    st.info("Downloading necessary NLTK data (punkt, stopwords)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    st.success("NLTK data downloaded.")

# --- Helper Functions ---


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


@st.cache_data(show_spinner=False)
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)


@st.cache_data(show_spinner=False)
def get_sentences(text):
    if not text:
        return []
    return sent_tokenize(text)


@st.cache_data(show_spinner=False)
def calculate_similarity(_text1, _text2):
    if not _text1 or not _text2:
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([_text1, _text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]
    except ValueError:
        return 0.0


def find_similar_sentences(sentences1, sentences2, threshold=0.5):
    similar_pairs = []
    if not sentences1 or not sentences2:
        return similar_pairs
    vectorizer = TfidfVectorizer()
    all_sentences = sentences1 + sentences2
    try:
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
        len1 = len(sentences1)
        similarity_matrix = cosine_similarity(
            tfidf_matrix[:len1], tfidf_matrix[len1:])
        for i in range(len(sentences1)):
            for j in range(len(sentences2)):
                if similarity_matrix[i, j] > threshold:
                    similar_pairs.append(
                        (sentences1[i], sentences2[j], similarity_matrix[i, j]))
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
    except ValueError:
        st.warning(
            "Could not compute similarity, possibly due to empty documents after preprocessing.")
        return []
    return similar_pairs


def find_differences(sentences1, sentences2, threshold=0.2):
    differences1 = []
    differences2 = []
    if not sentences1 or not sentences2:
        return sentences1, sentences2
    vectorizer = TfidfVectorizer()
    all_sentences = sentences1 + sentences2
    try:
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
        len1 = len(sentences1)
        similarity_matrix = cosine_similarity(
            tfidf_matrix[:len1], tfidf_matrix[len1:])
        for i in range(len(sentences1)):
            if similarity_matrix[i].max() < threshold:
                differences1.append(sentences1[i])
        similarity_matrix_t = similarity_matrix.T
        for j in range(len(sentences2)):
            if similarity_matrix_t[j].max() < threshold:
                differences2.append(sentences2[j])
    except ValueError:
        st.warning(
            "Could not compute differences, possibly due to empty documents after preprocessing.")
        return sentences1, sentences2
    return differences1, differences2


def generate_summary(text, num_sentences=5):
    sentences = get_sentences(text)
    if not sentences:
        return "No text to summarize."
    return "\n".join(sentences[:num_sentences])


def identify_keywords(text, keywords):
    found_keywords = {kw: [] for kw in keywords}
    sentences = get_sentences(text)
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            for sent in sentences:
                if kw.lower() in sent.lower():
                    found_keywords[kw].append(sent)
    return found_keywords


def cluster_sentences(sentences, num_clusters=5):
    if not sentences:
        return {}, "No sentences to cluster."

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(tfidf_matrix)

        clusters = defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(sentences[i])

        cluster_summary = f"Clustered into {num_clusters} groups based on content."
        return clusters, cluster_summary

    except ValueError:
        return {}, "Could not cluster sentences, possibly due to empty documents after preprocessing."


def visualize_clusters(clusters, sentences):
    if not clusters or not sentences:
        st.warning("No clusters or sentences to visualize.")
        return

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(tfidf_matrix.toarray())

        cluster_labels = []
        sentence_map = {}
        all_sentences = []
        label_index = 0
        for label, sents in clusters.items():
            cluster_labels.extend([f"Cluster {label}"] * len(sents))
            all_sentences.extend(sents)
            for i in range(len(sents)):
                sentence_map[all_sentences.index(sents[i])] = label

        df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
        df['Cluster'] = cluster_labels
        df['Sentence'] = all_sentences

        fig = px.scatter(df, x='PC1', y='PC2', color='Cluster', hover_data=['Sentence'],
                         title="Sentence Clusters Visualization")
        st.plotly_chart(fig)

    except ValueError:
        st.warning(
            "Could not visualize clusters, possibly due to issues with data dimensionality.")


# --- Keywords Definitions ---
KEYWORDS = {
    "Messaging Standards": ["MT format", "MX format", "ISO 20022", "FIN", "InterAct", "FileAct", "SWIFTNet"],
    "Security Protocols": ["encryption", "authentication", "HSM", "PKI", "CSP", "CSCF", "two-factor", "multi-factor", "security framework"],
    "Risk Mitigation": ["risk assessment", "fraud detection", "AML", "CTF", "sanctions screening", "KYC", "due diligence", "operational risk"],
    "Compliance Requirements": ["regulation", "directive", "compliance officer", "audit trail", "reporting obligation", "GDPR", "PSD2"],
    "Reporting Obligations": ["regulatory reporting", "transaction reporting", "suspicious activity report", "SAR", "audit log"],
    "Transaction Monitoring": ["monitoring system", "real-time monitoring", "transaction screening", "anomaly detection", "pattern analysis"]
}

# --- Streamlit UI ---
st.title("SWIFT Regulatory Compliance Analyzer")
st.subheader("Powered by Machine Learning for Document Comparison")

st.markdown(
    "Upload two or more SWIFT-related compliance documents (PDF or TXT) to analyze and compare them.")

# --- Sidebar for Uploads and Options ---
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Choose compliance documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload PDF or TXT files containing SWIFT compliance information."
    )

    st.header("⚙️ Analysis Options")
    similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.6, 0.05,
                                     help="Minimum cosine similarity score to consider sentences 'similar'.")
    difference_threshold = st.slider("Difference Threshold", 0.0, 0.5, 0.2, 0.05,
                                     help="Maximum cosine similarity score for a sentence to be considered 'different' from the other document.")
    summary_length = st.slider("Summary Sentences", 3, 10, 5,
                               1, help="Number of sentences for extractive summaries.")
    num_clusters = st.slider("Number of Clusters (for visualization)",
                             2, 10, 5, 1, help="Number of clusters to group sentences into.")

# --- Main Analysis Area ---
if uploaded_files and len(uploaded_files) >= 2:
    docs = {}
    doc_names = []
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                doc_name = uploaded_file.name
                doc_names.append(doc_name)
                content = uploaded_file.getvalue()
                raw_text = ""
                if uploaded_file.type == "application/pdf":
                    raw_text = extract_text_from_pdf(content)
                elif uploaded_file.type == "text/plain":
                    raw_text = content.decode("utf-8", errors='ignore')
                if raw_text:
                    docs[doc_name] = {
                        "raw": raw_text,
                        "processed": preprocess_text(raw_text),
                        "sentences": get_sentences(raw_text)
                    }
                else:
                    st.warning(f"Could not extract text from {doc_name}.")
                    docs[doc_name] = {"raw": "",
                                      "processed": "", "sentences": []}
    st.success(f"Processed {len(docs)} documents: {', '.join(doc_names)}")

    # --- Analysis Tabs ---
    tab_titles = [f"Comparison: {doc_names[0]} vs {doc_names[1]}",
                  "Summary & Gaps Analysis",
                  "Sentence Clustering",
                  "Sustainability Context",
                  "Intelligent Analysis (Gen AI)"]
    tabs = st.tabs(tab_titles)

    # --- Comparison Tab ---
    with tabs[0]:
        doc1_name = doc_names[0]
        doc2_name = doc_names[1]
        doc1 = docs.get(doc1_name)
        doc2 = docs.get(doc2_name)

        if doc1 and doc2:
            st.header(f"Comparing '{doc1_name}' and '{doc2_name}'")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Document 1: {doc1_name}")
                with st.expander("View Full Text", expanded=False):
                    st.text_area(
                        "Doc1 Raw", doc1['raw'], height=200, key="raw1")
            with col2:
                st.subheader(f"Document 2: {doc2_name}")
                with st.expander("View Full Text", expanded=False):
                    st.text_area(
                        "Doc2 Raw", doc2['raw'], height=200, key="raw2")

            st.divider()

            with st.spinner("Analyzing..."):
                overall_similarity = calculate_similarity(
                    doc1['processed'], doc2['processed'])
                similar_pairs = find_similar_sentences(
                    doc1['sentences'], doc2['sentences'], similarity_threshold)
                diff1, diff2 = find_differences(
                    doc1['sentences'], doc2['sentences'], difference_threshold)
                keywords1 = {cat: identify_keywords(
                    doc1['raw'], kws) for cat, kws in KEYWORDS.items()}
                keywords2 = {cat: identify_keywords(
                    doc2['raw'], kws) for cat, kws in KEYWORDS.items()}

            st.metric("Overall Document Similarity",
                      f"{overall_similarity:.2f}")
            st.markdown("---")

            st.subheader("Key Similarities")
            if similar_pairs:
                st.markdown(
                    f"Found **{len(similar_pairs)}** similar sentence pairs (similarity > {similarity_threshold}):")
                for i, (s1, s2, score) in enumerate(similar_pairs[:5]):
                    with st.expander(f"Pair {i+1} (Score: {score:.2f})", expanded=False):
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.info(f"**{doc1_name}:** {s1}")
                        with col_s2:
                            st.info(f"**{doc2_name}:** {s2}")
            else:
                st.info(
                    f"No significant sentence similarities found above the threshold ({similarity_threshold}).")

            st.subheader("Key Differences / Unique Content")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(
                    f"**Potentially Unique to '{doc1_name}'** (Low similarity < {difference_threshold} to '{doc2_name}')")
                if diff1:
                    with st.expander(f"Show {len(diff1)} sentences", expanded=False):
                        for sent in diff1:
                            st.warning(sent)
                else:
                    st.info(
                        "No significant unique sentences identified based on the threshold.")
            with col_d2:
                st.markdown(
                    f"**Potentially Unique to '{doc2_name}'** (Low similarity < {difference_threshold} to '{doc1_name}')")
                if diff2:
                    with st.expander(f"Show {len(diff2)} sentences", expanded=False):
                        for sent in diff2:
                            st.warning(sent)
                else:
                    st.info(
                        "No significant unique sentences identified based on the threshold.")

            st.divider()
            st.subheader("Keyword Analysis")
            for category, kws in KEYWORDS.items():
                with st.expander(f"Category: {category}", expanded=False):
                    cat_key1 = keywords1.get(category, {})
                    cat_key2 = keywords2.get(category, {})
                    common_kws = set(kw for kw, sents in cat_key1.items() if sents) & set(
                        kw for kw, sents in cat_key2.items() if sents)
                    unique_kws1 = set(kw for kw, sents in cat_key1.items(
                    ) if sents) - set(kw for kw, sents in cat_key2.items() if sents)
                    unique_kws2 = set(kw for kw, sents in cat_key2.items(
                    ) if sents) - set(kw for kw, sents in cat_key1.items() if sents)

                    if common_kws:
                        st.success(f"Common Keywords: {', '.join(common_kws)}")
                    if unique_kws1:
                        st.warning(
                            f"Keywords mainly in {doc1_name}: {', '.join(unique_kws1)}")
                    if unique_kws2:
                        st.warning(
                            f"Keywords mainly in {doc2_name}: {', '.join(unique_kws2)}")
                    if not common_kws and not unique_kws1 and not unique_kws2:
                        st.info(
                            "No keywords found in this category for either document.")
        else:
            st.warning(
                "Could not process one or both of the selected documents.")

    # --- Summary & Gaps Analysis Tab ---
    with tabs[1]:
        st.header("Summary & Gaps Analysis")
        st.markdown(
            "This section provides summaries and highlights potential gaps.")

        st.subheader("Document Summaries")
        for name, data in docs.items():
            if data['raw']:
                st.markdown(
                    f"**Summary for '{name}'** ({summary_length} sentences):")
                st.info(generate_summary(data['raw'], summary_length))
            else:
                st.info(f"Could not generate summary for '{name}'.")
        st.markdown("---")

        st.subheader("Potential Gaps (Keyword-Based)")
        st.markdown(
            "Highlights keywords present in some documents but not others.")
        all_keywords_found = {}
        for category, kws in KEYWORDS.items():
            all_keywords_found[category] = {}
            for kw in kws:
                present_in = [name for name, data in docs.items(
                ) if data['raw'] and kw.lower() in data['raw'].lower()]
                if present_in:
                    all_keywords_found[category][kw] = present_in

        gap_report = []
        num_docs = len(docs)
        for category, keywords_data in all_keywords_found.items():
            for keyword, present_in_docs in keywords_data.items():
                if 0 < len(present_in_docs) < num_docs:
                    missing_in_docs = list(
                        set(doc_names) - set(present_in_docs))
                    gap_report.append({
                        "Category": category,
                        "Keyword/Phrase": keyword,
                        "Present In": ", ".join(present_in_docs),
                        "Potentially Missing In": ", ".join(missing_in_docs)
                    })

        if gap_report:
            st.dataframe(pd.DataFrame(gap_report), use_container_width=True)
        else:
            st.info("No potential gaps detected based on the defined keywords.")
        st.markdown("---")
        st.markdown("**Note:** This analysis is based on the presence of specific keywords. A manual review is recommended for comprehensive gap identification.")

    # --- Sentence Clustering Tab ---
    with tabs[2]:
        st.header("Sentence Clustering")
        st.markdown("Group similar sentences from all documents.")

        all_sentences_from_docs = [
            sent for name, data in docs.items() for sent in data['sentences']]

        if all_sentences_from_docs:
            with st.spinner("Clustering..."):
                clusters, summary = cluster_sentences(
                    all_sentences_from_docs, num_clusters)
                st.info(summary)
                visualize_clusters(clusters, all_sentences_from_docs)
                st.subheader("Example Sentences in Clusters:")
                for label, sentences in clusters.items():
                    if sentences:
                        with st.expander(f"Cluster {label + 1}", expanded=False):
                            for sent in sentences[:5]:
                                st.write(f"- {sent}")
                            if len(sentences) > 5:
                                st.write("...")
        else:
            st.info("No sentences to cluster.")

    # --- Sustainability Context Tab ---
    with tabs[3]:
        st.header("Sustainability Context")
        st.markdown("""
        The analysis of SWIFT regulatory compliance plays a crucial role in fostering a more sustainable global financial system.
        """)
        st.subheader("Preventing Financial Crime that Harms Sustainability")
        st.markdown("""
        <span style="font-size:1.2em; color:{accent_color};">🚫</span> **Combating Illegal Activities:** Strong compliance helps prevent financial crime linked to unsustainable activities.
        """, unsafe_allow_html=True)
        st.subheader("Promoting Transparency and Accountability")
        st.markdown("""
        <span style="font-size:1.2em; color:{accent_color};">📈</span> **Enhanced Reporting:** Compliance often mandates transparent reporting, potentially including ESG factors.
        """, unsafe_allow_html=True)
        st.subheader(
            "Strengthening Financial Stability for Sustainable Development")
        st.markdown("""
        <span style="font-size:1.2em; color:{accent_color};">🛡️</span> **Resilient Systems:** SWIFT compliance contributes to a stable financial system, essential for sustainable investments.
        """, unsafe_allow_html=True)
        st.subheader("Fostering Ethical and Responsible Finance")
        st.markdown("""
        <span style="font-size:1.2em; color:{accent_color};">🤝</span> **Ethical Operations:** Adhering to regulations fosters a more ethical financial ecosystem for long-term sustainability.
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        This tool indirectly supports sustainability by aiding financial institutions in operating within a secure and transparent framework.
        """)

    # --- Intelligent Analysis (Gen AI) Tab ---
    with tabs[4]:
        st.header("Intelligent Analysis using Generative AI")
        st.markdown(
            "This section demonstrates potential integration with Generative AI for deeper insights.")

        if len(docs) >= 2:
            doc1_name = doc_names[0]
            doc2_name = doc_names[1]
            doc1_raw_text = docs[doc1_name]['raw']
            doc2_raw_text = docs[doc2_name]['raw']

            if doc1_raw_text and doc2_raw_text:
                st.subheader("Identify Potential Contradictions")
                st.markdown(
                    "Click the button below to analyze the documents for contradictory statements using a Generative AI model.")
                if st.button("Analyze for Contradictions"):
                    with st.spinner("Analyzing for contradictions using Gen AI..."):
                        # --- GEN AI INTEGRATION POINT ---
                        # **To implement this, you would:**
                        # 1. Choose a Gen AI platform (e.g., Google Cloud AI, OpenAI).
                        # 2. Install the platform's Python client library (e.g., `google-generativeai`).
                        # 3. Set up authentication and get your API key.
                        # 4. Initialize the Gen AI model.
                        # 5. Construct a prompt that asks the model to identify contradictions, for example:
                        #    `prompt = f"Analyze the following two documents and identify any statements that contradict each other regarding SWIFT compliance: Document 1: {doc1_raw_text} Document 2: {doc2_raw_text}"`
                        # 6. Send the prompt to the Gen AI model and get the response.
                        # 7. Display the model's response here.

                        st.info(
                            "Contradiction analysis using a Generative AI model would be displayed here. This requires integration with a Gen AI platform's API.")

                st.subheader("Generate Comparative Summary")
                st.markdown(
                    "Click the button to generate a summary highlighting the key similarities and differences between the documents using Gen AI.")
                if st.button("Generate Comparative Summary"):
                    with st.spinner("Generating comparative summary using Gen AI..."):
                        # --- GEN AI INTEGRATION POINT ---
                        # **To implement this, you would:**
                        # 1. Construct a prompt asking for a comparative summary:
                        #    `prompt = f"Generate a summary highlighting the key similarities and differences in the SWIFT compliance requirements outlined in the following two documents: Document 1: {doc1_raw_text} Document 2: {doc2_raw_text}"`
                        # 2. Send the prompt to the Gen AI model and display the response.

                        st.info(
                            "A comparative summary generated by a Generative AI model would be shown here.")

                st.subheader("Ask a Comparative Question")
                question = st.text_input(
                    "Enter your question about the documents (e.g., 'What are the key differences in security protocols?'):")
                if st.button("Ask Question"):
                    with st.spinner("Answering your question using Gen AI..."):
                        # --- GEN AI INTEGRATION POINT ---
                        # **To implement this, you would:**
                        # 1. Construct a prompt that includes the question and the document content:
                        #    `prompt = f"Answer the following question based on these two documents: '{question}' Document 1: {doc1_raw_text} Document 2: {doc2_raw_text}"`
                        # 2. Send the prompt to the Gen AI model and display the answer.

                        st.info(
                            "The answer to your question from a Generative AI model would appear here.")
            else:
                st.warning(
                    "Please ensure both documents have extracted text for intelligent analysis.")
        else:
            st.warning(
                "Please upload at least two documents for intelligent analysis.")

elif uploaded_files and len(uploaded_files) < 2:
    st.warning("⚠️ Please upload at least two documents for comparison.")
else:
    st.info("✨ Upload two or more documents using the sidebar to begin analysis.")

# Add a footer
st.markdown("---")
st.caption("SWIFT Compliance Analyzer v0.2 - For informational purposes only.")
