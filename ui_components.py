# import streamlit as st
# import pandas as pd
# import openai
# from typing import Dict, Any, List

# from analysis import retrieve_relevant_context, cluster_sentences # Assuming cluster function is in analysis.py
# from utils import KEYWORDS # Assuming KEYWORDS is in utils.py

# import plotly.express as px
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction.text import TfidfVectorizer # Needed for visualize_clusters

# # The large generate_impact_report_content function would also go here or in a separate reporting.py

# def generate_rag_prompt(user_query: str, context: str) -> List[Dict]:
#     """Constructs the messages payload for the OpenAI API with RAG context."""
#     system_message = (
#         "You are an expert AI assistant specializing in SWIFT regulatory compliance. "
#         "Your task is to answer the user's question based *only* on the provided context from the documents. "
#         "If the answer is not in the context, state that clearly. Do not use external knowledge."
#     )
    
#     prompt = (
#         f"{context}\n\n"
#         f"User Question: {user_query}\n\n"
#         "Please answer the user's question using only the provided context."
#     )
    
#     return [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": prompt}
#     ]

# def render_rag_tab(rag_retriever, all_sentences: List[str]):
#     """Renders the Intelligent Analysis (RAG) tab."""
#     st.header("Intelligent Analysis using Retrieval-Augmented Generation (RAG)")
#     st.markdown(
#         "Ask a specific question about the uploaded documents. The AI will retrieve the most relevant sections and use them to generate a grounded answer."
#     )

#     try:
#         openai.api_type = "azure"
#         openai.api_key = st.secrets["azure"]["api_key"]
#         openai.api_base = st.secrets["azure"]["endpoint"]
#         openai.api_version = st.secrets["azure"]["api_version"]
#         deployment = st.secrets["azure"]["chat_deployment"]
#         openai_available = True
#     except (KeyError, Exception) as e:
#         st.warning(f"Azure OpenAI is not configured correctly in secrets.toml. RAG feature disabled. Error: {e}")
#         openai_available = False

#     if not openai_available:
#         return

#     user_query = st.text_input(
#         "Ask a question about the documents:",
#         "What are the main requirements for transaction monitoring?"
#     )

#     if st.button("Get AI-Powered Answer"):
#         if not user_query.strip():
#             st.warning("Please enter a question.")
#             return

#         with st.spinner("Retrieving context and generating answer..."):
#             # 1. Retrieve context using RAG
#             context = retrieve_relevant_context(user_query, all_sentences, rag_retriever)
            
#             with st.expander("View Retrieved Context (What the AI is using)"):
#                 st.info(context)

#             # 2. Generate prompt and call OpenAI
#             messages = generate_rag_prompt(user_query, context)
            
#             try:
#                 response = openai.ChatCompletion.create(
#                     engine=deployment,
#                     messages=messages,
#                     temperature=0.2, # Lower temperature for more factual, less creative answers
#                     max_tokens=1000
#                 )
#                 ai_response = response.choices[0].message.content
#                 st.subheader("AI Generated Answer:")
#                 st.markdown(ai_response)
#             except Exception as e:
#                 st.error(f"An error occurred with the OpenAI API: {e}")

# # INSIDE ui_components.py, EDIT this function

# def render_main_ui(docs: Dict, results: Dict, rag_retriever):
#     """Renders all the analysis tabs."""
#     doc_names = list(docs.keys())
    
#     # Add "Sentence Clustering" to the tab titles
#     tab_titles = [
#         f"Comparison: {doc_names[0]} vs {doc_names[1]}",
#         "Gaps & Summary",
#         "Sentence Clustering", # <-- ADDED
#         "Intelligent Q&A (RAG)"
#     ]
#     tabs = st.tabs(tab_titles)

#     with tabs[0]:
#         # ... (comparison tab code remains the same)
#         st.header(f"Comparing '{doc_names[0]}' and '{doc_names[1]}'")
#         st.metric("Overall Document Similarity", f"{results['overall_similarity']:.2f}")

#     with tabs[1]:
#         # ... (gaps & summary tab code remains the same)
#         st.header("Gaps Analysis")
#         if results['gap_report']:
#             st.dataframe(pd.DataFrame(results['gap_report']), use_container_width=True)
#         else:
#             st.info("No potential gaps detected.")

#     with tabs[2]: # <-- THIS IS THE NEW CLUSTERING TAB
#         num_clusters = st.session_state.get('num_clusters', 5) # Get value from sidebar
#         render_clustering_tab(results['all_sentences'], num_clusters)

#     with tabs[3]: # <-- NOTE THE INDEX CHANGE
#         render_rag_tab(rag_retriever, results['all_sentences'])
        
# # PASTE THESE TWO FUNCTIONS INTO ui_components.py

# def visualize_clusters(clusters: dict, sentences: list):
#     """Renders a Plotly scatter plot of the sentence clusters."""
#     if not clusters or not sentences or len(sentences) < 2:
#         st.warning("Not enough data to visualize clusters.")
#         return

#     vectorizer = TfidfVectorizer()
#     try:
#         tfidf_matrix = vectorizer.fit_transform(sentences).toarray()
#         if tfidf_matrix.shape[1] < 2:
#             st.warning("Cannot visualize clusters: Not enough distinct terms in the documents for PCA.")
#             return

#         pca = PCA(n_components=2, random_state=42)
#         reduced_features = pca.fit_transform(tfidf_matrix)

#         cluster_labels = []
#         all_sents_in_clusters = []
#         for label, sents in clusters.items():
#             cluster_labels.extend([f"Cluster {label + 1}"] * len(sents))
#             all_sents_in_clusters.extend(sents)

#         # Create DataFrame from the sentences that were actually clustered
#         df = pd.DataFrame()
#         df['PC1'] = reduced_features[:len(all_sents_in_clusters), 0]
#         df['PC2'] = reduced_features[:len(all_sents_in_clusters), 1]
#         df['Cluster'] = cluster_labels
#         df['Sentence'] = all_sents_in_clusters

#         fig = px.scatter(
#             df, x='PC1', y='PC2', color='Cluster',
#             hover_data={'Sentence': True, 'PC1': False, 'PC2': False},
#             title="Sentence Clusters Visualization (2D Projection using PCA)"
#         )
#         fig.update_traces(marker=dict(size=10, opacity=0.8))
#         st.plotly_chart(fig, use_container_width=True)

#     except Exception as e:
#         st.error(f"An error occurred during cluster visualization: {e}")


# def render_clustering_tab(all_sentences: list, num_clusters: int):
#     """Renders the entire UI for the sentence clustering tab."""
#     st.header("Sentence Clustering")
#     st.markdown("Group similar sentences from all documents to identify recurring themes.")

#     if not all_sentences:
#         st.info("No sentences available to cluster.")
#         return
    
#     with st.spinner("Clustering sentences..."):
#         # Call the analysis function
#         clusters, summary = cluster_sentences(all_sentences, num_clusters)
#         st.success(summary)

#         # Call the visualization function
#         visualize_clusters(clusters, all_sentences)

#         if clusters:
#             st.subheader("Explore Clusters")
#             for label in sorted(clusters.keys()):
#                 sentences_in_cluster = clusters[label]
#                 with st.expander(f"Cluster {label + 1} ({len(sentences_in_cluster)} sentences)"):
#                     for sent in sentences_in_cluster[:5]: # Show top 5
#                         st.write(f"- {sent}")
#                     if len(sentences_in_cluster) > 5:
#                         st.write("...")