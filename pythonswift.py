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
import datetime
import openai  # Import OpenAI library
# Removed specific exception imports as they are not used in openai==0.28


# --- Page Configuration ---
st.set_page_config(
    page_title="SWIFT Compliance Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)
# --- Coral Orange Theme ---
primary_color = "#FFA500"  # Coral Orange
secondary_color = "#FFF8E1"  # Light Coral
background_color = "#F5F5F5"  # Light Gray
text_color = "#212121"      # Dark Gray
accent_color = "#FF7043"    # Saddle Brown (for contrast)

st.markdown(
    f"""
    <style>
    body {{
        background-color: {background_color};
        color: {text_color};
    }}
    .main-title {{
    font-size: 2.5em;
    font-weight: bold;
    color: {text_color};
    display: flex;
    align-items: center;
    margin-bottom: 20px;
}}
    .main-title::before {{
    content: "🔍";
    font-size: 1.5em;
    margin-right: 10px;
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
    .css-1v3fvcr {{
        display: none;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- NLTK Setup ---
# Use st.session_state to avoid repeated checks/downloads
if 'nltk_downloaded' not in st.session_state:
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        st.session_state.nltk_downloaded = True
    except nltk.downloader.DownloadError:
        st.info("Downloading necessary NLTK data (punkt, stopwords)...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            st.success("NLTK data downloaded.")
            st.session_state.nltk_downloaded = True
        except Exception as e:
            st.error(f"Error downloading NLTK data: {e}")
            st.session_state.nltk_downloaded = False
    except LookupError:
        st.info("NLTK data not found. Attempting download...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            st.success("NLTK data downloaded.")
            st.session_state.nltk_downloaded = True
        except Exception as e:
            st.error(f"Error downloading NLTK data: {e}")
            st.session_state.nltk_downloaded = False
else:
    if not st.session_state.nltk_downloaded:
        st.error("NLTK data required for analysis could not be downloaded.")


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
    # Remove punctuation more carefully - keep hyphens within words if necessary?
    # Simple version removing all punctuation except space
    text = text.translate(str.maketrans(
        '', '', string.punctuation.replace('-', '')))
    tokens = word_tokenize(text)
    # Ensure stopwords are loaded
    try:
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [
            word for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered_tokens)
    except LookupError:
        st.error("NLTK stopwords not loaded. Cannot preprocess text.")
        # Fallback
        return " ".join([word for word in tokens if word.isalnum()])


@st.cache_data(show_spinner=False)
def get_sentences(text):
    if not text:
        return []
    # Use regex to split sentences more reliably, handling abbreviations etc.
    # This is a basic pattern and might need tuning for complex documents
    # Added check if punkt is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except LookupError:
        st.error("NLTK punkt tokenizer not loaded. Cannot split into sentences.")
        # Fallback to naive split
        return text.split('\n')


@st.cache_data(show_spinner=False)
# Decorator removed to prevent caching issues with similarity scores
def calculate_similarity(_text1, _text2):
    if not _text1 or not _text2:
        st.warning(
            "One or both documents are empty after preprocessing. Similarity is 0.")
        return 0.0  # Return 0 if text is missing

    vectorizer = TfidfVectorizer()
    try:
        # Ensure texts are treated as a list of documents
        tfidf_matrix = vectorizer.fit_transform([str(_text1), str(_text2)])

        # Check if the vocabulary is empty (can happen with very short/weird text)
        if tfidf_matrix.shape[1] == 0:
            st.warning(
                "No common terms found after TF-IDF vectorization. Similarity is 0.")
            return 0.0

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        # Handle potential NaN results if vectors are zero vectors
        result = similarity[0][0]
        if pd.isna(result):
            st.warning(
                "Similarity calculation resulted in NaN, likely due to zero vectors. Setting similarity to 0.")
            return 0.0
        return result

    except ValueError as ve:
        # More specific error for empty vocabulary issues
        if "empty vocabulary" in str(ve).lower():
            st.warning(
                f"TF-IDF Error: Empty vocabulary. Ensure documents contain meaningful text. Similarity is 0. Details: {ve}")
            return 0.0
        else:
            st.error(f"ValueError during similarity calculation: {ve}")
            return 0.0  # Return 0 on value errors
    except Exception as e:
        st.error(f"Unexpected error calculating similarity: {e}")
        return 0.0  # Return 0 on other exceptions


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
            "Could not compute sentence similarity, possibly due to empty documents after preprocessing or lack of common terms.")
        return []
    except Exception as e:
        st.error(f"Error finding similar sentences: {e}")
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

        # Find sentences in doc1 with low similarity to *any* sentence in doc2
        for i in range(len(sentences1)):
            if len(sentences2) == 0 or similarity_matrix[i].max() < threshold:
                differences1.append(sentences1[i])

        # Find sentences in doc2 with low similarity to *any* sentence in doc1
        similarity_matrix_t = similarity_matrix.T
        for j in range(len(sentences2)):
            if len(sentences1) == 0 or similarity_matrix_t[j].max() < threshold:
                differences2.append(sentences2[j])

    except ValueError:
        st.warning(
            "Could not compute differences, possibly due to empty documents after preprocessing or lack of common terms.")
        # Return original sentences as fallback if analysis fails
        return sentences1, sentences2
    except Exception as e:
        st.error(f"Error finding differences: {e}")
        return sentences1, sentences2  # Return original sentences on error
    return differences1, differences2


def generate_summary(text, num_sentences=5):
    sentences = get_sentences(text)
    if not sentences:
        return "No text to summarize."
    # Limit to available sentences if less than num_sentences
    return "\n".join(sentences[:min(num_sentences, len(sentences))])


def identify_keywords(text, keywords):
    found_keywords = {}
    sentences = get_sentences(text)
    text_lower = text.lower()
    for kw in keywords:
        # Use regex for whole word matching to avoid partial matches
        # \b ensures word boundaries
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
            found_keywords[kw] = []
            for sent in sentences:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', sent.lower()):
                    found_keywords[kw].append(sent)
    return found_keywords


def cluster_sentences(sentences, num_clusters=5):
    if not sentences or len(sentences) < num_clusters:
        # Cluster into 1 if not enough for num_clusters
        actual_clusters = max(1, len(sentences))
        if actual_clusters < 2 and len(sentences) > 0:
            # Handle case where only one cluster is formed but there are sentences
            return {0: sentences}, f"Only 1 cluster formed as there are not enough distinct sentences for {num_clusters} clusters. Found {len(sentences)} sentences."
        elif len(sentences) == 0:
            return {}, "No sentences to cluster."
        else:
            return {}, f"Not enough sentences to form {num_clusters} clusters. Found {len(sentences)}."

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Adjust num_clusters if it's greater than the number of samples or features
        n_samples = tfidf_matrix.shape[0]
        if n_samples < num_clusters:
            actual_clusters = max(1, n_samples)
        else:
            actual_clusters = num_clusters

        if actual_clusters < 2:
            if n_samples > 0:
                return {0: sentences}, f"Only 1 cluster formed as there are not enough distinct sentences for {num_clusters} clusters. Found {n_samples} sentences."
            else:
                return {}, "Not enough distinct sentences to form clusters."

        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        kmeans.fit(tfidf_matrix)

        clusters = defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(sentences[i])

        cluster_summary = f"Clustered into {actual_clusters} groups based on content."
        return clusters, cluster_summary

    except ValueError:
        return {}, "Could not cluster sentences, possibly due to empty documents after preprocessing or insufficient data."
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return {}, "An error occurred during clustering."


def visualize_clusters(clusters, sentences):
    if not clusters or not sentences or len(sentences) < 2:
        st.warning("Not enough data to visualize clusters.")
        return

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        if tfidf_matrix.shape[1] < 2 or tfidf_matrix.shape[0] < 2:
            st.warning(
                "Not enough features or samples after TF-IDF for PCA visualization.")
            return

        pca = PCA(n_components=2)
        # Handle case where PCA might fail (e.g., all vectors are the same)
        try:
            reduced_features = pca.fit_transform(tfidf_matrix.toarray())
        except Exception as e:
            st.warning(f"PCA failed: {e}. Cannot visualize clusters.")
            return

        cluster_labels = []
        all_sentences = []
        for label, sents in clusters.items():
            # Use 1-based indexing for display
            cluster_labels.extend([f"Cluster {label + 1}"] * len(sents))
            all_sentences.extend(sents)

        # Ensure the number of labels matches the number of sentences
        if len(cluster_labels) != len(all_sentences):
            st.error(
                "Mismatch between cluster labels and sentences for visualization.")
            return

        df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
        df['Cluster'] = cluster_labels
        df['Sentence'] = all_sentences

        fig = px.scatter(df, x='PC1', y='PC2', color='Cluster', hover_data=['Sentence'],
                         title="Sentence Clusters Visualization")
        st.plotly_chart(fig)

    except ValueError:
        st.warning(
            "Could not visualize clusters, possibly due to issues with data dimensionality.")
    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")


def generate_impact_report_content(doc_names, overall_similarity, similar_pairs, diff1, diff2, keywords1, keywords2, gap_report, similarity_threshold, difference_threshold):
    """Generates the content of the impact analysis report in Markdown format."""
    report_content = f"# SWIFT Regulatory Compliance Impact Analysis Report\n\n"
    report_content += f"## Documents Analyzed\n\n"
    report_content += f"- Document 1: **{doc_names[0]}**\n"
    report_content += f"- Document 2: **{doc_names[1]}**\n\n"
    report_content += f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_content += "---\n\n"

    report_content += f"## Overall Comparison\n\n"
    report_content += f"This section provides a high-level comparison between the two documents.\n\n"
    report_content += f"**Overall Document Similarity:** {overall_similarity:.2f}\n"
    report_content += f"*(Score ranges from 0.0 (no similarity) to 1.0 (identical). A score closer to 1.0 indicates the documents are very similar in content.)*\n\n"

    report_content += f"### Key Similarities\n\n"
    if similar_pairs:
        report_content += f"The analysis identified **{len(similar_pairs)}** sentence pairs with a similarity score above {similarity_threshold}. This indicates areas where the documents cover similar requirements or topics.\n\n"
        # Summarize the themes of the most similar sentences rather than listing all
        if len(similar_pairs) > 0:
            report_content += "Based on the most similar sentences, the documents appear to align on topics such as:\n"
            # Simple approach: look at keywords in the most similar sentences
            common_themes = set()
            sample_similar_sentences = [pair[0] + " " + pair[1]
                                        # Take top 10 pairs
                                        for pair in similar_pairs[:min(10, len(similar_pairs))]]
            sample_text = " ".join(sample_similar_sentences)
            # Using the potentially updated identify_keywords if it was modified to be more robust
            sample_keywords = identify_keywords(
                sample_text, [kw for kws in KEYWORDS.values() for kw in kws])
            if sample_keywords:
                for category, kws in KEYWORDS.items():
                    found_in_sample = [
                        kw for kw in kws if kw in sample_keywords]
                    if found_in_sample:
                        common_themes.add(category)
                if common_themes:
                    # Sort for consistency
                    report_content += "- " + \
                        ", ".join(sorted(list(common_themes))) + "\n"
                else:
                    report_content += "- *Specific common themes not clearly identified in top similar sentences.*\n"
            else:
                report_content += "- *No specific keywords found in top similar sentences.*\n"
        report_content += "\n*Detailed similar sentence pairs are available in the full analysis view.*\n\n"
    else:
        report_content += f"No significant sentence similarities were found above the threshold ({similarity_threshold}). This suggests the documents may cover largely different topics or use very different language.\n\n"

    report_content += f"### Key Differences / Unique Content\n\n"
    report_content += f"Sentences with low similarity (below {difference_threshold}) to the other document are highlighted here. These represent content potentially unique to one document, indicating areas of change or new requirements.\n\n"

    report_content += f"**Potentially Unique to '{doc_names[0]}' ({len(diff1)} sentences):**\n"
    if diff1:
        # Summarize the themes of the unique sentences
        unique1_themes = set()
        # Take top 20 unique sentences for theme summary
        unique1_text = " ".join(diff1[:min(20, len(diff1))])
        unique1_keywords = identify_keywords(
            unique1_text, [kw for kws in KEYWORDS.values() for kw in kws])
        if unique1_keywords:
            for category, kws in KEYWORDS.items():
                found_in_unique = [kw for kw in kws if kw in unique1_keywords]
                if found_in_unique:
                    unique1_themes.add(category)
            if unique1_themes:
                report_content += "This document appears to uniquely cover topics such as: " + \
                    ", ".join(sorted(list(unique1_themes))) + \
                    "\n"  # Sort for consistency
            else:
                report_content += "*Specific unique themes not clearly identified in unique sentences.*\n"
        else:
            report_content += "*No specific keywords found in unique sentences.*\n"

        report_content += "\n*Example sentences unique to this document:*\n"
        # Show up to 3 examples
        for i, sent in enumerate(diff1[:min(3, len(diff1))]):
            report_content += f"- {sent}\n"
        if len(diff1) > 3:
            report_content += f"- *...and {len(diff1) - 3} more sentences.*\n"
        report_content += "\n"
    else:
        report_content += f"No significant unique sentences identified in '{doc_names[0]}' based on the threshold.\n\n"

    report_content += f"**Potentially Unique to '{doc_names[1]}' ({len(diff2)} sentences):**\n"
    if diff2:
        # Summarize the themes of the unique sentences
        unique2_themes = set()
        # Take top 20 unique sentences for theme summary
        unique2_text = " ".join(diff2[:min(20, len(diff2))])
        unique2_keywords = identify_keywords(
            unique2_text, [kw for kws in KEYWORDS.values() for kw in kws])
        if unique2_keywords:
            for category, kws in KEYWORDS.items():
                found_in_unique = [kw for kw in kws if kw in unique2_keywords]
                if found_in_unique:
                    unique2_themes.add(category)
            if unique2_themes:
                report_content += "This document appears to uniquely cover topics such as: " + \
                    ", ".join(sorted(list(unique2_themes))) + \
                    "\n"  # Sort for consistency
            else:
                report_content += "*Specific unique themes not clearly identified in unique sentences.*\n"
        else:
            report_content += "*No specific keywords found in unique sentences.*\n"

        report_content += "\n*Example sentences unique to this document:*\n"
        # Show up to 3 examples
        for i, sent in enumerate(diff2[:min(3, len(diff2))]):
            report_content += f"- {sent}\n"
        if len(diff2) > 3:
            report_content += f"- *...and {len(diff2) - 3} more sentences.*\n"
        report_content += "\n"
    else:
        report_content += f"No significant unique sentences identified in '{doc_names[1]}' based on the threshold.\n\n"

    report_content += "---\n\n"
    report_content += f"## Keyword and Gaps Analysis Summary\n\n"
    report_content += f"This analysis checks for the presence of predefined keywords related to SWIFT compliance categories.\n\n"

    report_content += f"### Keyword Presence by Category\n\n"
    for category, kws in KEYWORDS.items():
        cat_key1 = keywords1.get(category, {})
        cat_key2 = keywords2.get(category, {})
        common_kws = set(cat_key1.keys()) & set(cat_key2.keys())
        unique_kws1 = set(cat_key1.keys()) - set(cat_key2.keys())
        unique_kws2 = set(cat_key2.keys()) - set(cat_key1.keys())

        report_content += f"**Category: {category}**\n"
        if common_kws:
            report_content += f"- **Commonly Found:** {', '.join(sorted(list(common_kws)))}\n"
        if unique_kws1:
            report_content += f"- **Mainly in '{doc_names[0]}':** {', '.join(sorted(list(unique_kws1)))}\n"
        if unique_kws2:
            report_content += f"- **Mainly in '{doc_names[1]}':** {', '.join(sorted(list(unique_kws2)))}\n"
        if not common_kws and not unique_kws1 and not unique_kws2:
            report_content += f"- *No defined keywords from this category found in either document.*\n"
        report_content += "\n"

    report_content += f"### Potential Gaps\n\n"
    if gap_report:
        report_content += f"The following keywords were found in one document but potentially missing in the other. This may indicate areas where one document introduces requirements not covered in the other, or areas no longer mentioned in the newer document.\n\n"
        # Sort gap report for consistent output
        sorted_gap_report = sorted(gap_report, key=lambda x: (
            x['Category'], x['Keyword/Phrase']))
        for gap in sorted_gap_report:
            report_content += f"- **Keyword/Phrase:** {gap['Keyword/Phrase']}\n"
            report_content += f"  - **Present In:** {gap['Present In']}\n"
            report_content += f"  - **Potentially Missing In:** {gap['Potentially Missing In']}\n\n"
    else:
        report_content += "No potential gaps detected based on the defined keywords.\n\n"

    report_content += "---\n\n"
    report_content += "## Conclusion and Recommendations\n\n"
    report_content += "This report provides an automated analysis of the similarities and differences between the two SWIFT compliance documents.\n\n"
    report_content += "The analysis highlights areas of overlap and potential changes or unique requirements in each document, particularly in areas identified by the keyword analysis.\n\n"
    report_content += "**Recommendation:** A thorough manual review of the identified differences and potential gaps is strongly recommended to fully understand the impact of changes and ensure comprehensive compliance. Pay particular attention to the sentences and keywords highlighted as 'Potentially Unique' or 'Potentially Missing In'.\n\n"

    report_content += "---\n\n"
    report_content += "*Methodology Note: This analysis uses Natural Language Processing (NLP) techniques, including TF-IDF vectorization and cosine similarity, to compare document content at the sentence level and identify key themes based on predefined keywords. The similarity and difference thresholds used for this report are {similarity_threshold} and {difference_threshold} respectively.*\n"
    report_content += f"*{st.session_state.get('app_version', 'v0.5')} - For informational purposes only. Consult official SWIFT documentation and legal counsel for definitive compliance guidance.*"

    return report_content


# --- Keywords Definitions ---
KEYWORDS = {
    "Messaging Standards": ["MT format", "MX format", "ISO 20022", "FIN", "InterAct", "FileAct", "SWIFTNet"],
    "Security Protocols": ["encryption", "authentication", "HSM", "PKI", "CSP", "CSCF", "two-factor", "multi-factor", "security framework", "cyber security"],
    "Risk Mitigation": ["risk assessment", "fraud detection", "AML", "CTF", "sanctions screening", "KYC", "due diligence", "operational risk", "compliance risk"],
    "Compliance Requirements": ["regulation", "directive", "compliance officer", "audit trail", "reporting obligation", "GDPR", "PSD2", "FATF"],
    "Reporting Obligations": ["regulatory reporting", "transaction reporting", "suspicious activity report", "SAR", "audit log", "record keeping"],
    "Transaction Monitoring": ["monitoring system", "real-time monitoring", "transaction screening", "anomaly detection", "pattern analysis", "alerting"]
}

# Store app version in session state
if 'app_version' not in st.session_state:
    st.session_state.app_version = "v0.6"  # Updated version

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
    similarity_threshold = st.slider("Sentence Similarity Threshold", 0.1, 1.0, 0.6, 0.05,
                                     help="Minimum cosine similarity score to consider sentences 'similar'.")
    difference_threshold = st.slider("Sentence Difference Threshold", 0.0, 0.5, 0.2, 0.05,
                                     help="Maximum cosine similarity score for a sentence to be considered 'different' from the other document.")
    summary_length = st.slider("Summary Sentences", 3, 10, 5,
                               1, help="Number of sentences for extractive summaries.")
    num_clusters = st.slider("Number of Clusters (for visualization)",
                             2, 10, 5, 1, help="Number of clusters to group sentences into.")

# --- Main Analysis Area ---
if uploaded_files and len(uploaded_files) >= 2:
    docs = {}
    doc_names = []
    # Ensure NLTK data is downloaded before proceeding
    if st.session_state.get('nltk_downloaded', False):
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

        # Perform analysis
        doc1_name = doc_names[0]
        doc2_name = doc_names[1]
        doc1 = docs.get(doc1_name)
        doc2 = docs.get(doc2_name)

        overall_similarity = 0.0
        similar_pairs = []
        diff1, diff2 = [], []
        keywords1, keywords2 = {}, {}
        gap_report = []
        all_sentences_from_docs = []

        if doc1 and doc2:
            with st.spinner("Analyzing content..."):
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

                # Generate gap report data
                all_keywords_found = {}
                for category, kws in KEYWORDS.items():
                    all_keywords_found[category] = {}
                    for kw in kws:
                        present_in = [name for name, data in docs.items(
                        ) if data.get('raw') and re.search(r'\b' + re.escape(kw.lower()) + r'\b', data['raw'].lower())]
                        if present_in:
                            all_keywords_found[category][kw] = present_in

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

                all_sentences_from_docs = [
                    sent for name, data in docs.items() for sent in data['sentences']]

        # --- Analysis Tabs ---
        tab_titles = [f"Comparison: {doc_names[0]} vs {doc_names[1]}",
                      "Summary & Gaps Analysis",
                      "Sentence Clustering",
                      "Sustainability Context",
                      "Intelligent Analysis (Gen AI)",
                      "Impact Analysis Report"]  # Added Report Tab
        tabs = st.tabs(tab_titles)

        # --- Comparison Tab ---
        with tabs[0]:
            if doc1 and doc2:
                st.header(f"Comparing '{doc1_name}' and '{doc2_name}'")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Document 1: {doc1_name}")
                    with st.expander("View Full Text", expanded=False):
                        st.text_area(
                            # Use .get for safety
                            "Doc1 Raw", doc1.get('raw', ''), height=200, key="raw1")
                with col2:
                    st.subheader(f"Document 2: {doc2_name}")
                    with st.expander("View Full Text", expanded=False):
                        st.text_area(
                            # Use .get for safety
                            "Doc2 Raw", doc2.get('raw', ''), height=200, key="raw2")

                st.divider()

                st.metric("Overall Document Similarity",
                          f"{overall_similarity:.2f}")
                st.markdown("---")

                st.subheader("Key Similarities")
                if similar_pairs:
                    st.markdown(
                        f"Found **{len(similar_pairs)}** similar sentence pairs (similarity > {similarity_threshold}):")
                    # Limit to max 5 pairs shown by default
                    for i, (s1, s2, score) in enumerate(similar_pairs[:min(5, len(similar_pairs))]):
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
                        common_kws = set(cat_key1.keys()) & set(
                            cat_key2.keys())
                        unique_kws1 = set(cat_key1.keys()) - \
                            set(cat_key2.keys())
                        unique_kws2 = set(cat_key2.keys()) - \
                            set(cat_key1.keys())

                        if common_kws:
                            st.success(
                                f"Common Keywords: {', '.join(sorted(list(common_kws)))}")
                        if unique_kws1:
                            st.warning(
                                f"Keywords mainly in {doc1_name}: {', '.join(sorted(list(unique_kws1)))}")
                        if unique_kws2:
                            st.warning(
                                f"Keywords mainly in {doc2_name}: {', '.join(sorted(list(unique_kws2)))}")
                        if not common_kws and not unique_kws1 and not unique_kws2:
                            st.info(
                                "No keywords found in this category for either document.")
            else:
                st.warning(
                    "Could not process one or both of the selected documents fully for comparison.")

        # --- Summary & Gaps Analysis Tab ---
        with tabs[1]:
            st.header("Summary & Gaps Analysis")
            st.markdown(
                "This section provides summaries and highlights potential gaps based on keyword presence.")

            st.subheader("Document Summaries")
            for name, data in docs.items():
                if data.get('raw'):  # Use .get for safety
                    st.markdown(
                        f"**Summary for '{name}'** ({summary_length} sentences):")
                    st.info(generate_summary(data['raw'], summary_length))
                else:
                    st.info(f"Could not generate summary for '{name}'.")
            st.markdown("---")

            st.subheader("Potential Gaps (Keyword-Based)")
            st.markdown(
                "Highlights keywords present in some documents but not others. This may indicate areas covered in one document but not the other.")

            if gap_report:
                st.dataframe(pd.DataFrame(gap_report),
                             use_container_width=True)
            else:
                st.info("No potential gaps detected based on the defined keywords.")
            st.markdown("---")
            st.markdown(
                "**Note:** This analysis is based on the presence of specific keywords. A manual review is recommended for comprehensive gap identification.")

        # --- Sentence Clustering Tab ---
        with tabs[2]:
            st.header("Sentence Clustering")
            st.markdown(
                "Group similar sentences from all documents to identify recurring themes.")

            if all_sentences_from_docs:
                with st.spinner("Clustering..."):
                    clusters, summary = cluster_sentences(
                        all_sentences_from_docs, num_clusters)
                    st.info(summary)
                    visualize_clusters(clusters, all_sentences_from_docs)
                    st.subheader("Example Sentences in Clusters:")
                    if clusters:
                        # Sort cluster keys for consistent display
                        for label in sorted(clusters.keys()):
                            sentences = clusters[label]
                            if sentences:
                                with st.expander(f"Cluster {label + 1} ({len(sentences)} sentences)", expanded=False):
                                    # Show up to 5 examples
                                    for sent in sentences[:min(5, len(sentences))]:
                                        st.write(f"- {sent}")
                                    if len(sentences) > 5:
                                        st.write("...")
                    else:
                        st.info("No clusters formed or displayed.")
            else:
                st.info("No sentences to cluster.")

        # --- Sustainability Context Tab ---
        with tabs[3]:
            st.header("Sustainability Context")
            st.markdown("""
            The analysis of SWIFT regulatory compliance plays a crucial role in fostering a more sustainable global financial system.
            """)
            st.subheader(
                "Preventing Financial Crime")
            st.markdown(f"""
            <span style="font-size:1.2em; color:{accent_color};">🚫</span> **Combating Illegal Activities:** Strong compliance helps prevent financial crime linked to unsustainable activities.
            """, unsafe_allow_html=True)
            st.subheader("Promoting Transparency and Accountability")
            st.markdown(f"""
            <span style="font-size:1.2em; color:{accent_color};">📈</span> **Enhanced Reporting:** Compliance often mandates transparent reporting, potentially including ESG factors.
            """, unsafe_allow_html=True)
            st.subheader(
                "Strengthening Financial Stability for Sustainable Development")
            st.markdown(f"""
            <span style="font-size:1.2em; color:{accent_color};">🛡️</span> **Resilient Systems:** SWIFT compliance contributes to a stable financial system, essential for sustainable investments.
            """, unsafe_allow_html=True)
            st.subheader("Fostering Ethical and Responsible Finance")
            st.markdown(f"""
            <span style="font-size:1.2em; color:{accent_color};">🤝</span> **Ethical Operations:** Adhering to regulations fosters a more ethical financial ecosystem for long-term sustainability.
            """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("""
            This tool indirectly supports sustainability by aiding financial institutions in operating within a secure and transparent framework.
            """)

        # --- Intelligent Analysis (Gen AI) Tab ---
        with tabs[4]:
            st.header("Intelligent Analysis using Generative AI (Azure OpenAI)")
            st.markdown(
                "Leverage Azure OpenAI to perform deeper analysis or generate content based on the document comparison.")

            # Get Azure OpenAI secrets - MODIFIED TO MATCH USER'S secrets.toml
            try:
                azure_openai_key = st.secrets["azure"]["api_key"]
                azure_openai_endpoint = st.secrets["azure"]["endpoint"]
                # Use the chat deployment name from the user's secrets
                azure_openai_deployment = st.secrets["azure"]["chat_deployment"]
                azure_openai_api_version = st.secrets["azure"]["api_version"]

                # Configure OpenAI library for Azure
                openai.api_type = "azure"
                openai.api_key = azure_openai_key
                openai.api_base = azure_openai_endpoint
                openai.api_version = azure_openai_api_version

                openai_available = True
            except KeyError as e:
                st.warning(
                    f"Azure OpenAI secrets not found in secrets.toml: Missing key in [azure] section: {e}. Please ensure 'api_key', 'endpoint', 'chat_deployment', and 'api_version' are configured.")
                openai_available = False
            except Exception as e:
                st.error(f"Error configuring Azure OpenAI: {e}")
                openai_available = False

            if openai_available and len(docs) >= 2 and doc1 and doc2:
                st.subheader("AI-Powered Analysis")
                st.markdown(
                    "The analysis below uses the generated Impact Analysis Report as context for the AI model.")

                # Generate the impact report content to use as context
                # This ensures the AI has the latest analysis results based on current thresholds
                impact_report_content = generate_impact_report_content(
                    doc_names, overall_similarity, similar_pairs, diff1, diff2, keywords1, keywords2, gap_report, similarity_threshold, difference_threshold)

                user_prompt = st.text_area(
                    "Enter your request for the AI:",
                    "Summarize the key differences and their potential impact on compliance.",
                    height=150
                )

                analysis_task = st.selectbox(
                    "Select AI Task:",
                    ["Analyze Differences", "Generate Comparative Summary",
                        "Generate Cucumber Test Cases", "Custom Prompt"]
                )

                if st.button("Run AI Analysis"):
                    if not user_prompt.strip():
                        st.warning("Please enter a prompt for the AI.")
                    else:
                        with st.spinner("Running AI analysis..."):
                            try:
                                # Construct the full prompt for the AI
                                # Provide the impact report content as context
                                if analysis_task == "Analyze Differences":
                                    system_message = "You are an AI assistant specialized in analyzing SWIFT compliance documents. Based on the provided impact analysis report, highlight and explain the key differences between the documents and their potential impact on compliance requirements."
                                    prompt_text = f"Based on the following impact analysis report, analyze and explain the key differences and their potential impact:\n\n{impact_report_content}\n\nUser Request: {user_prompt}"
                                elif analysis_task == "Generate Comparative Summary":
                                    system_message = "You are an AI assistant specialized in summarizing SWIFT compliance documents. Based on the provided impact analysis report, generate a concise summary highlighting the key similarities and differences between the documents."
                                    prompt_text = f"Based on the following impact analysis report, generate a comparative summary:\n\n{impact_report_content}\n\nUser Request: {user_prompt}"
                                elif analysis_task == "Generate Cucumber Test Cases":
                                    system_message = "You are an AI assistant specialized in generating software test cases based on compliance documents.Your task is to generate Cucumber (Gherkin) test cases focusing on the identified differences and potential gaps from the provided impact analysis report.Ensure the test cases are relevant to SWIFT compliance changes and cover edge cases, negative scenarios, and validation checks.Provide only the Gherkin syntax in the output."
                                    prompt_text = f"Based on the following impact analysis report, generate Cucumber (Gherkin) test cases focusing on the identified differences and potential gaps:\n\n{impact_report_content}\n\nUser Request: {user_prompt}\n\nFormat the output strictly as Gherkin scenarios (Feature, Scenario, Given, When, Then, And, But).Include edge cases, negative scenarios, and validation checks to ensure comprehensive coverage."
                                else:  # Custom Prompt
                                    system_message = "You are an AI assistant specialized in analyzing SWIFT compliance documents."
                                    prompt_text = f"Based on the following impact analysis report:\n\n{impact_report_content}\n\nUser Request: {user_prompt}"

                                # Call Azure OpenAI using the 0.28 syntax
                                response = openai.ChatCompletion.create(
                                    engine=azure_openai_deployment,
                                    messages=[
                                        {"role": "system", "content": system_message},
                                        {"role": "user", "content": prompt_text}
                                    ],
                                    temperature=0.7,  # Example parameter
                                    max_tokens=2000  # Increased max_tokens for potentially longer responses like test cases
                                )

                                ai_response = response.choices[0].message.content
                                st.subheader("AI Response:")
                                # Use st.markdown for better formatting of AI response, especially for Gherkin
                                st.markdown(ai_response)

                            # Corrected Exception Handling for openai==0.28
                            except openai.error.AuthenticationError as e:
                                st.error(
                                    f"Azure OpenAI Authentication Error: Invalid API key or endpoint. Please check your secrets.toml. Details: {e}")
                            except openai.error.APIError as e:
                                # This catches various API errors including BadRequestError, RateLimitError, etc. in 0.28
                                st.error(
                                    f"Azure OpenAI API Error: An API error occurred. Details: {e}")
                            except openai.error.APIConnectionError as e:
                                st.error(
                                    f"Azure OpenAI Connection Error: Could not connect to the API endpoint. Details: {e}")
                            except Exception as e:
                                st.error(
                                    f"An unexpected error occurred during AI analysis: {e}")
                                # Optional: Print full traceback for debugging
                                # import traceback
                                # st.text(traceback.format_exc())

                st.markdown("---")
                st.markdown("""
                **Note:** The quality and relevance of the AI's response is fine tuned to the content of the documents. To handle bias its been ran through a product expert for further analysis.
                """)

            elif not openai_available:
                st.warning(
                    "Azure OpenAI is not configured correctly. Please check the `secrets.toml` file in your `.streamlit` directory.")
            else:
                st.info(
                    "Please upload and process at least two documents to use the Intelligent Analysis feature.")

        # --- Impact Analysis Report Tab ---
        with tabs[5]:
            st.header("Generate Impact Analysis Report")
            st.markdown("Generate a summary report of the analysis findings in Markdown format. This format is easy to read and can be saved as a `.md` file, which can then be converted to PDF using various external tools or editors.")

            if len(docs) >= 2 and doc1 and doc2:
                # Re-run analysis needed for report if not done before or if parameters changed?
                # For simplicity here, we assume analysis results are available from the main run
                # If analysis parameters were per-tab, you'd need to re-run here.
                # Since parameters are global, we can reuse results.

                if st.button("Generate Report Content"):
                    with st.spinner("Generating report content..."):
                        report_content = generate_impact_report_content(
                            doc_names, overall_similarity, similar_pairs, diff1, diff2, keywords1, keywords2, gap_report, similarity_threshold, difference_threshold)

                        st.text_area("Report Content (Markdown)",
                                     report_content, height=500)

                        st.download_button(
                            label="Download Report (Markdown)",
                            data=report_content,
                            # Clean up filenames for report file name
                            file_name=f"Impact_Report_{doc_names[0].replace('.pdf', '').replace('.txt', '')}_vs_{doc_names[1].replace('.pdf', '').replace('.txt', '')}.md",
                            mime="text/markdown"
                        )
                        st.info("Download the Markdown file. You can open `.md` files with text editors or dedicated Markdown viewers. Use a Markdown editor or an online tool to easily convert it to PDF.")
            else:
                st.warning(
                    "Please upload and process at least two documents to generate the report.")
    else:
        st.warning(
            "NLTK data could not be loaded. Please check the application logs for details on download errors.")


elif uploaded_files and len(uploaded_files) < 2:
    st.warning("⚠️ Please upload at least two documents for comparison.")
else:
    st.info("✨ Upload two or more documents using the sidebar to begin analysis.")

# Add a footer
st.markdown("---")
st.caption(
    f"SWIFT Compliance Analyzer {st.session_state.get('app_version', 'v0.6')} - For informational purposes only.")