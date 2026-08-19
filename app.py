import streamlit as st
import time

print("Startup: Importing libraries...")
# Page Config (First Streamlit command)
st.set_page_config(page_title="AG News Intelligence Hub", layout="wide")

import pandas as pd
import plotly.express as px
from src.data_loader import load_ag_news_data
from src.models import train_svm_model, load_svm_model, save_model, HybridClassifier, load_transformer_model
from src.topic_model import (
    fit_lda_model, fit_nmf_model, get_top_words, get_document_topics, 
    get_dominant_topics, get_topic_stats, get_topic_samples, 
    compute_topic_similarity, plot_topic_similarity, suggest_topic_name,
    compute_topic_diversity, compute_topic_purity,
    get_lda_vis, get_nmf_vis
)
from src.eda import plot_length_distributions, get_extreme_articles, detect_length_outliers, get_statistical_summary, get_outlier_proportions, get_category_proportions
from src.utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer

print("Startup: Libraries imported.")

# Custom CSS for aesthetics (Clean Black & Gray High-Contrast Theme)
# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown, p, span, label, li {
        font-family: 'Outfit', sans-serif;
        color: #000000 !important;
    }

    /* Main Background - Clean Soft Gray */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Sidebar - Crisp White */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Headings - Pure Black */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 700;
    }
    
    /* Custom Card Style for Buttons - Clean Grey */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #9E9E9E 0%, #757575 100%);
        color: #FFFFFF !important;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        font-weight: 600;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #BDBDBD 0%, #9E9E9E 100%);
        color: #FFFFFF !important;
    }

    /* Primary Button - Light Blue (Classify) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #64B5F6 0%, #42A5F5 100%) !important;
        color: #FFFFFF !important;
        border: none;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #90CAF9 0%, #64B5F6 100%) !important;
        color: #FFFFFF !important;
    }

    /* Inputs and Text Areas */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 10px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        color: #000000 !important;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 5px solid #000000;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 10px 20px;
        border: 1px solid #E0E0E0;
        color: #000000 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #F0F0F0 !important;
        color: #000000 !important;
        font-weight: bold;
        border: 1px solid #000000;
    }

    /* Dropdown / Selectbox - Grey Theme */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {
        background-color: #F0F0F0 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 10px;
        color: #333333 !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #999999 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #666666 !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #333333 !important;
    }

    /* Dropdown popover & menu container */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] > div > div {
        background-color: #F5F5F5 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 10px;
    }

    /* Dropdown menu items */
    ul[data-baseweb="menu"],
    ul[role="listbox"],
    div[data-baseweb="menu"] {
        background-color: #F5F5F5 !important;
    }

    ul[data-baseweb="menu"] li,
    ul[role="listbox"] li,
    li[role="option"],
    div[data-baseweb="menu"] li {
        background-color: #F5F5F5 !important;
        color: #333333 !important;
    }

    ul[data-baseweb="menu"] li:hover,
    ul[role="listbox"] li:hover,
    li[role="option"]:hover,
    div[data-baseweb="menu"] li:hover {
        background-color: #E0E0E0 !important;
    }

    li[aria-selected="true"] {
        background-color: #E0E0E0 !important;
        color: #333333 !important;
    }

    /* File Uploader - Grey Theme */
    section[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #9E9E9E 0%, #757575 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        font-weight: 600;
    }

    section[data-testid="stFileUploader"] button:hover,
    div[data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #BDBDBD 0%, #9E9E9E 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
        background-color: #F5F5F5 !important;
        border: 2px dashed #BDBDBD !important;
        border-radius: 10px;
    }

    /* Expander / Analysis Settings - Grey Theme */
    div[data-testid="stExpander"] summary {
        background-color: #F0F0F0 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 10px;
        color: #333333 !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background-color: #E0E0E0 !important;
    }

    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary p {
        color: #333333 !important;
    }

    div[data-testid="stExpander"] summary svg {
        fill: #666666 !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #CCCCCC !important;
        border-radius: 10px;
    }
    
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #000000; margin-bottom: 2rem;'>
        📰 AG News Intelligence Hub
    </h1>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def get_data():
    return load_ag_news_data()

try:
    with st.spinner("Loading AG News Dataset..."):
        train_df, test_df = get_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/news.png") # Placeholder icon
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Module", 
    ["Exploratory Analysis", "Text Classification", "Topic Modeling"])

label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
label_names = list(label_map.values())

# --- Models ---
@st.cache_resource
def get_svm():
    model = load_svm_model()
    if model is None:
        model = train_svm_model(train_df)
        save_model(model)
    return model

@st.cache_resource
def get_hybrid():
    svm = get_svm()
    return HybridClassifier(svm)

@st.cache_resource
def get_transformer():
    return load_transformer_model()

@st.cache_data
def get_model_metrics(_model, model_name, _test_data):
    # Use a subset for speed if dataset is large, or full if feasible. 
    # AG News test is 7600, might take a few seconds for Hybrid loop. 
    # Using small subset for responsiveness as planned.
    subset_size = 200 if model_name == "Transformer" else 1000 # Transformer is slower
    subset = _test_data.sample(n=subset_size, random_state=42) if len(_test_data) > subset_size else _test_data
    y_true = subset['label']
    
    if model_name == "SVM":
        y_pred_idx = _model.predict(subset['text'])
    elif model_name == "Transformer":
        y_pred_idx = _model.predict(subset['text'].tolist())
    else: # Hybrid
        y_pred_idx = _model.predict(subset['text'].tolist())
        
    acc = accuracy_score(y_true, y_pred_idx)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_idx, average='weighted', zero_division=0)
    
    return {"Accuracy": acc, "Precision": p, "Recall": r, "F1": f1}

# --- 1. Exploratory Analysis ---
if mode == "Exploratory Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Deep dive into article lengths and complexity.")
    
    # Preprocessing for EDA
    with st.spinner("Calculating length statistics..."):
        train_df['char_count'] = train_df['text'].astype(str).apply(len)
        train_df['word_count'] = train_df['text'].astype(str).apply(lambda x: len(x.split()))
        
        avg_words = train_df['word_count'].mean()
        avg_chars = train_df['char_count'].mean()
        
    st.markdown("### 📈 Key Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Avg Words / Article", f"{avg_words:.1f}")
    m2.metric("Avg Characters / Article", f"{avg_chars:.1f}")
    
    tab1, tab2, tab3 = st.tabs(["Statistical Summary", "Length Distributions", "Extreme Articles"])
    
    with tab1:
        st.subheader("Statistical Summary")
        st.markdown("Detailed statistics for Text Lengths and Word Counts.")
        
        # 1. Comprehensive Stats Table
        stats_df = get_statistical_summary(train_df)
        st.write("#### Text Complexity Stats (Mean, Median, Mode, Variance)")
        st.dataframe(stats_df, use_container_width=True)
        
        # 2. Category Distribution
        st.write("#### Category Distribution")
        cat_prop_df = get_category_proportions(train_df)
        fig_cat_pie = px.pie(cat_prop_df, values='Count', names='Category',
                             title="Overall Category Distribution",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_cat_pie, use_container_width=True)
        
        # 3. Outlier Proportions
        st.write("#### Outlier Analysis")
        
        # Calculate totals
        outliers_raw = detect_length_outliers(train_df)
        outlier_prop_df = get_outlier_proportions(train_df)
        
        st.info(f"Detected **{len(outliers_raw)}** outliers out of **{len(train_df)}** total articles.")
        
        col_prop1, col_prop2 = st.columns(2)
        
        # Pie Chart First
        with col_prop1:
            fig_pie = px.pie(outlier_prop_df, values='Outlier Count', names='Category', 
                             title="Outlier Distribution by Category",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Table Second
        with col_prop2:
            st.caption("Outlier Proportions (Relative to Total Outliers)")
            st.dataframe(outlier_prop_df, hide_index=True)

    with tab2:
        st.subheader("Distribution Analysis")
        fig_char, fig_word = plot_length_distributions(train_df)
        
        c1, c2 = st.columns(2)
        if fig_word:
            c1.plotly_chart(fig_word, use_container_width=True)
        if fig_char:
            c2.plotly_chart(fig_char, use_container_width=True)
            
    with tab3:
        st.subheader("Longest & Shortest Articles")
        longest, shortest = get_extreme_articles(train_df)

        # Display tables
        st.write("**Top 5 Longest Articles**")
        st.table(longest)
        st.write("**Top 5 Shortest Articles**")
        st.table(shortest)


# --- 2. Text Classification ---
elif mode == "Text Classification":
    st.header("🔍 Text Classification")
    st.markdown("Classify news articles using machine learning and rule-based systems.")
    
    analyzer_tab1, analyzer_tab2 = st.tabs(["Single Text Analysis", "Batch Processing"])
    
    # --- Single Text ---
    with analyzer_tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            article_text = st.text_area("Enter Article Text", height=200, 
                                       placeholder="Type or paste a news snippet here...")
        
        with col2:
            model_options = ["SVM (Pure ML)", "Hybrid (Rules + SVM)", "Transformer (BERT)", "Compare All"]
            model_choice = st.selectbox("Choose System", model_options)
            classify_btn = st.button("Classify Now", type="primary")

        if classify_btn and article_text:
            
            # Helper to predict
            def run_predict(model_type, text):
                start_time = time.time()
                if model_type == "SVM":
                    m = get_svm()
                    idx = m.predict([text])[0]
                    desc = "Standard SVM"
                elif model_type == "Transformer":
                    m = get_transformer()
                    idx = m.predict(text)[0]
                    desc = "BERT Transformer"
                else: # Hybrid
                    m = get_hybrid()
                    idx = m.predict(text)[0]
                    desc = "Hybrid Rules+SVM"
                end_time = time.time()
                latency = end_time - start_time
                return label_map[idx], desc, latency

            if model_choice == "Compare All":
                # Run all 3
                col_res1, col_res2, col_res3 = st.columns(3)
                
                # SVM
                pred_svm, _, lat_svm = run_predict("SVM", article_text)
                with col_res1:
                    st.info(f"**SVM:** {pred_svm}")
                    st.caption(f"Lat: {lat_svm:.4f}s")
                
                # Hybrid
                pred_hyb, _, lat_hyb = run_predict("Hybrid", article_text)
                with col_res2:
                    st.warning(f"**Hybrid:** {pred_hyb}")
                    st.caption(f"Lat: {lat_hyb:.4f}s")
                    
                # Transformer
                pred_trans, _, lat_trans = run_predict("Transformer", article_text)
                with col_res3:
                    st.success(f"**Transformer:** {pred_trans}")
                    st.caption(f"Lat: {lat_trans:.4f}s")
                    
                # Automatic Metrics Display (General Performance)
                st.subheader("Model Evaluation (Test Set Performance)")
                st.caption("Note: Transformer metrics calculated on smaller subset for speed.")
                
                with st.spinner("Calculating performance metrics for all models..."):
                    svm_metrics = get_model_metrics(get_svm(), "SVM", test_df)
                    hyb_metrics = get_model_metrics(get_hybrid(), "Hybrid", test_df)
                    trans_metrics = get_model_metrics(get_transformer(), "Transformer", test_df)
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown("##### SVM")
                    st.write(f"**Accuracy:** {svm_metrics['Accuracy']:.4f}")
                    st.write(f"F1: {svm_metrics['F1']:.4f}")
                with m2:
                    st.markdown("##### Hybrid")
                    st.write(f"**Accuracy:** {hyb_metrics['Accuracy']:.4f}")
                    st.write(f"F1: {hyb_metrics['F1']:.4f}")
                with m3:
                    st.markdown("##### Transformer")
                    st.write(f"**Accuracy:** {trans_metrics['Accuracy']:.4f}")
                    st.write(f"F1: {trans_metrics['F1']:.4f}")

                # Comparative Graphs
                st.markdown("---")
                st.subheader("Comparative Analysis")
                
                g1, g2 = st.columns(2)
                
                pastel_colors = ['#90CAF9', '#FFCC80', '#A5D6A7'] # Blue, Orange, Green
                
                models_list = ['SVM', 'Hybrid', 'Transformer']
                acc_list = [svm_metrics['Accuracy'], hyb_metrics['Accuracy'], trans_metrics['Accuracy']]
                lat_list = [lat_svm, lat_hyb, lat_trans]
                
                # 1. Accuracy
                acc_df = pd.DataFrame({'Model': models_list, 'Accuracy': acc_list})
                fig_acc = px.bar(acc_df, x='Model', y='Accuracy', color='Model', 
                                  title="Accuracy Comparison", range_y=[0, 1.1],
                                  color_discrete_sequence=pastel_colors, template="plotly_white")
                fig_acc.update_layout(showlegend=False)
                g1.plotly_chart(fig_acc, use_container_width=True)
                
                # 2. Latency
                lat_df = pd.DataFrame({'Model': models_list, 'Latency (s)': lat_list})
                fig_lat = px.bar(lat_df, x='Model', y='Latency (s)', color='Model',
                                 title="Inference Latency (Lower is Better)",
                                 color_discrete_sequence=pastel_colors, template="plotly_white")
                fig_lat.update_layout(showlegend=False)
                g2.plotly_chart(fig_lat, use_container_width=True)
                
            else:
                if "Transformer" in model_choice:
                    sys_name = "Transformer"
                elif "SVM" in model_choice:
                    sys_name = "SVM" 
                else: 
                    sys_name = "Hybrid"
                    
                pred, info, lat = run_predict(sys_name, article_text)
                st.success(f"**Category:** {pred}")
                st.caption(f"Latency: {lat:.4f} seconds | {info}")
                
                # Show General Metrics for single model too
                st.markdown("### Model Performance (General)")
                curr_model = get_svm() if sys_name == "SVM" else (get_hybrid() if sys_name == "Hybrid" else get_transformer())
                metrics = get_model_metrics(curr_model, sys_name, test_df)
                
                c_met1, c_met2, c_met3, c_met4 = st.columns(4)
                c_met1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                c_met2.metric("Precision", f"{metrics['Precision']:.4f}")
                c_met3.metric("Recall", f"{metrics['Recall']:.4f}")
                c_met4.metric("F1 Score", f"{metrics['F1']:.4f}")

    # --- Batch Processing ---
    with analyzer_tab2:
        st.markdown("Upload a CSV file. If it contains a 'label' column (0-3), metrics will be calculated.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file:
            df_batch = pd.read_csv(uploaded_file)
            if 'text' in df_batch.columns:
                st.write(f"Loaded {len(df_batch)} rows.")
                
                batch_model = st.radio("Select Model(s)", ["SVM", "Hybrid", "Transformer", "Compare All"])
                
                if st.button("Process Batch"):
                    results = {}
                    times = {}
                    
                    if batch_model == "Compare All":
                         models_to_run = ["SVM", "Hybrid", "Transformer"]
                    elif batch_model == "Both": # Maintaining backward compatibility if needed, else remove
                         models_to_run = ["SVM", "Hybrid"] 
                    else:
                         models_to_run = [batch_model]
                    
                    for m_name in models_to_run:
                        with st.spinner(f"Running {m_name}..."):
                            start_batch = time.time()
                            if m_name == "SVM":
                                preds_idx = get_svm().predict(df_batch['text'])
                            elif m_name == "Transformer":
                                preds_idx = get_transformer().predict(df_batch['text'].tolist())
                            else:
                                preds_idx = get_hybrid().predict(df_batch['text'].tolist())
                            end_batch = time.time()
                            times[m_name] = end_batch - start_batch
                            
                            df_batch[f'pred_{m_name}'] = [label_map[p] for p in preds_idx]
                            results[m_name] = preds_idx

                    st.write("### Predictions Preview")
                    st.dataframe(df_batch.head())
                    
                    # Latency Display
                    st.write("### Batch Latency")
                    for m_name, t in times.items():
                        st.write(f"**{m_name}:** {t:.4f} seconds")
                    
                    # Metrics Calculation if Ground Truth exists
                    gt_col = next((c for c in df_batch.columns if c.lower() in ['label', 'class', 'category', 'target']), None)
                    
                    if gt_col:
                        st.subheader("📈 Performance Metrics")
                        metrics_data = []
                        
                        for m_name in models_to_run:
                            y_true = df_batch[gt_col]
                            y_pred = results[m_name]
                            
                            acc = accuracy_score(y_true, y_pred)
                            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                            
                            st.write(f"**{m_name} Results:**")
                            st.write(f"Accuracy: {acc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
                            
                            metrics_data.append([m_name, acc, p, r, f1])
                        
                        if batch_model == "Compare All" or batch_model == "Both":
                            met_df = pd.DataFrame(metrics_data, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
                            
                            pastel_colors = ['#90CAF9', '#FFCC80', '#A5D6A7'] if len(met_df) > 2 else ['#90CAF9', '#FFCC80']
                            
                            st.subheader("Comparative Analysis")
                            
                            g_batch1, g_batch2 = st.columns(2)
                            
                            # 1. Accuracy Comparison
                            fig_acc = px.bar(met_df, x="Model", y="Accuracy", color="Model",
                                         title="Model Comparison (Accuracy)",
                                         range_y=[0, 1.1],
                                         color_discrete_sequence=pastel_colors,
                                         template="plotly_white")
                            fig_acc.update_layout(showlegend=False)
                            g_batch1.plotly_chart(fig_acc, use_container_width=True)
                            
                            # 2. Latency Comparison
                            lat_batch_df = pd.DataFrame(list(times.items()), columns=['Model', 'Latency (s)'])
                            fig_lat = px.bar(lat_batch_df, x="Model", y="Latency (s)", color="Model",
                                             title="Batch Latency Comparison",
                                             color_discrete_sequence=pastel_colors,
                                             template="plotly_white")
                            fig_lat.update_layout(showlegend=False)
                            g_batch2.plotly_chart(fig_lat, use_container_width=True)
                            
                            # Also show full metrics table
                            st.write("DETAILED METRICS:")
                            st.dataframe(met_df)
                    else:
                        st.warning("No ground truth label column found. Metrics cannot be calculated.")

                    st.download_button("Download Results", df_batch.to_csv(index=False), "results.csv", "text/csv")
            else:
                st.error("CSV must contain a 'text' column.")

# --- 2. Exploratory Analysis ---


elif mode == 'Topic Modeling':
    st.header('💡 Topic Modeling')
    st.markdown('Discover hidden themes across **127,600** news articles using Topic Modeling.')
    
    # 1. Configuration
    with st.expander("Analysis Settings", expanded=True):
        col_set1, col_set2, col_set3 = st.columns(3)
        with col_set1:
            algo_choice = st.selectbox('Algorithm', ['LDA (Gibbs-like Online VB)', 'NMF (Non-Negative Matrix Factorization)', 'Compare Both'])
        with col_set2:
            num_topics = st.slider('Number of Topics', 5, 40, 10 if algo_choice == 'Compare Both' else 20)
        with col_set3:
            subset_size = st.number_input('Data Subset Size for Extraction', 1000, len(train_df), 2000 if algo_choice == 'Compare Both' else 5000, step=1000)
            st.caption("Larger subsets improve quality but increase processing time.")

    if st.button('🚀 Extract Themes & Build Dashboard'):
        subset_df = train_df.sample(subset_size, random_state=42).copy()
        texts_list = subset_df['text'].tolist()
        
        if algo_choice != 'Compare Both':
            is_lda = 'LDA' in algo_choice
            model_label = 'LDA' if is_lda else 'NMF'
            
            with st.spinner(f'Fitting {model_label} model on {subset_size} articles...'):
                start_time = time.time()
                if is_lda:
                    model, vectorizer, feature_matrix, preprocessed_texts = fit_lda_model(texts_list, n_topics=num_topics)
                else:
                    model, vectorizer, feature_matrix, preprocessed_texts = fit_nmf_model(texts_list, n_topics=num_topics)
                fit_duration = time.time() - start_time
                
            st.success(f"Successfully fitted {model_label} in {fit_duration:.2f} seconds!")
            
            # Extract topics and metrics
            topics_keyword_map = get_top_words(model, vectorizer.get_feature_names_out(), 10)
            doc_topic_dist = get_document_topics(model, feature_matrix)
            dominant_topics = get_dominant_topics(doc_topic_dist)
            topic_stats_df = get_topic_stats(dominant_topics, num_topics)
            
            # Heuristic naming
            topic_names = []
            for i in range(num_topics):
                t_id = f"Topic {i+1}"
                kws = topics_keyword_map[t_id]
                name = suggest_topic_name(kws)
                topic_names.append(name)
            topic_stats_df['Topic Name'] = topic_names
            
            topic_samples = get_topic_samples(subset_df, dominant_topics, n_samples=2)
            topic_diversity = compute_topic_diversity(model, vectorizer.get_feature_names_out(), n_top_words=10)
            topic_purity = compute_topic_purity(dominant_topics, subset_df['label']) if 'label' in subset_df.columns else None
            
            # --- Results Dashboard ---
            dash_tab1, dash_tab2, dash_tab3 = st.tabs(["Topic Distribution", "Detailed Explorer", "Interactive pyLDAvis Visualization"])
            
            with dash_tab1:
                st.subheader("📊 Article Distribution & Overview")
                topic_stats_df['Display Label'] = topic_stats_df['Topic'] + ": " + topic_stats_df['Topic Name']
                fig_counts = px.bar(topic_stats_df, x='Display Label', y='Count', 
                                    text='Count', color='Count',
                                    title=f"Articles per Topic under {model_label}",
                                    labels={'Display Label': 'Topic (Primary Keywords)'},
                                    color_continuous_scale='Viridis')
                st.plotly_chart(fig_counts, use_container_width=True)
                
                st.dataframe(topic_stats_df[['Topic', 'Topic Name', 'Count', 'Percentage']], 
                             use_container_width=True, hide_index=True)
                             
            with dash_tab2:
                st.subheader("🕵️ Detailed Theme Explorer")
                for i in range(num_topics):
                    topic_id = f"Topic {i+1}"
                    keywords = topics_keyword_map[topic_id]
                    samples = topic_samples.get(topic_id, ["No samples available for this topic."])
                    topic_display_name = topic_names[i]
                    topic_count = topic_stats_df[topic_stats_df['Topic'] == topic_id]['Count'].values[0]
                    
                    with st.expander(f"📌 {topic_id} ({topic_count} articles) | Theme: {topic_display_name}", expanded=False):
                        st.write(f"**Keywords:** `{', '.join(keywords)}`")
                        st.markdown("**Sample Articles:**")
                        for s_ptr, sample_text in enumerate(samples):
                            st.info(f"Sample {s_ptr+1}: {sample_text[:400]}...")
                            
            with dash_tab3:
                st.subheader("🌐 pyLDAvis Interactive Topic Landscape")
                st.markdown("Use this interactive visualization to analyze inter-topic distance (left panel) and term relevance / frequency (right panel).")
                with st.spinner("Preparing interactive pyLDAvis visualizer..."):
                    import pyLDAvis
                    if is_lda:
                        vis_data = get_lda_vis(model, feature_matrix, vectorizer)
                    else:
                        # Prepare count matrix for NMF pyLDAvis
                        count_vec = CountVectorizer(max_df=0.95, min_df=2, vocabulary=vectorizer.vocabulary_)
                        tf_matrix_nmf = count_vec.fit_transform(preprocessed_texts)
                        vis_data = get_nmf_vis(model, tf_matrix_nmf, count_vec, doc_topic_dist)
                    
                    html_string = pyLDAvis.prepared_data_to_html(vis_data)
                
                import streamlit.components.v1 as components
                components.html(html_string, height=850, width=1300, scrolling=True)
                
        else:
            with st.spinner("Fitting LDA and NMF models..."):
                # Fit LDA
                start_lda = time.time()
                lda_model, lda_vec, lda_tf, lda_prep = fit_lda_model(texts_list, n_topics=num_topics)
                time_lda = time.time() - start_lda
                
                # Fit NMF
                start_nmf = time.time()
                nmf_model, nmf_vec, nmf_tfidf, nmf_prep = fit_nmf_model(texts_list, n_topics=num_topics)
                time_nmf = time.time() - start_nmf
                
            st.success(f"Models fitted successfully! LDA took {time_lda:.2f}s | NMF took {time_nmf:.2f}s")
            
            # 1. Performance Metric Comparison
            col_met1, col_met2 = st.columns(2)
            col_met1.metric("LDA Training Time", f"{time_lda:.3f} s")
            col_met2.metric("NMF Training Time", f"{time_nmf:.3f} s", delta=f"{time_nmf - time_lda:.3f} s" if time_nmf < time_lda else f"+{time_nmf - time_lda:.3f} s", delta_color="inverse")
            
            # Extract topic details for both
            lda_keywords = get_top_words(lda_model, lda_vec.get_feature_names_out(), 10)
            lda_dist = get_document_topics(lda_model, lda_tf)
            lda_dominant = get_dominant_topics(lda_dist)
            lda_stats = get_topic_stats(lda_dominant, num_topics)
            lda_diversity = compute_topic_diversity(lda_model, lda_vec.get_feature_names_out(), n_top_words=10)
            lda_purity = compute_topic_purity(lda_dominant, subset_df['label']) if 'label' in subset_df.columns else None
            
            nmf_keywords = get_top_words(nmf_model, nmf_vec.get_feature_names_out(), 10)
            nmf_dist = get_document_topics(nmf_model, nmf_tfidf)
            nmf_dominant = get_dominant_topics(nmf_dist)
            nmf_stats = get_topic_stats(nmf_dominant, num_topics)
            nmf_diversity = compute_topic_diversity(nmf_model, nmf_vec.get_feature_names_out(), n_top_words=10)
            nmf_purity = compute_topic_purity(nmf_dominant, subset_df['label']) if 'label' in subset_df.columns else None
            
            # Heuristic names
            lda_names = [suggest_topic_name(lda_keywords[f"Topic {i+1}"]) for i in range(num_topics)]
            nmf_names = [suggest_topic_name(nmf_keywords[f"Topic {i+1}"]) for i in range(num_topics)]
            
            lda_stats['Topic Name'] = lda_names
            nmf_stats['Topic Name'] = nmf_names
            
            # Comparitive Dashboard Tabs
            tab_comp1, tab_comp2, tab_comp3 = st.tabs(["Algorithm Comparison", "LDA Landscape (pyLDAvis)", "NMF Landscape (pyLDAvis)"])
            
            with tab_comp1:
                st.subheader("📈 Algorithm Performance & Topic Profile Comparison")
                
                metrics_data = {
                    'Metric': ['Topic Diversity', 'Topic Purity'],
                    'LDA': [f"{lda_diversity:.3f}", f"{lda_purity:.3f}" if lda_purity is not None else 'N/A'],
                    'NMF': [f"{nmf_diversity:.3f}", f"{nmf_purity:.3f}" if nmf_purity is not None else 'N/A']
                }
                metrics_table = pd.DataFrame(metrics_data)
                st.markdown("#### Topic Quality Comparison")
                st.table(metrics_table)
                
                # Plot side-by-side bar chart of document counts per topic
                lda_stats['Algorithm'] = 'LDA'
                nmf_stats['Algorithm'] = 'NMF'
                combined_stats = pd.concat([lda_stats, nmf_stats])
                
                fig_comp = px.bar(combined_stats, x='Topic', y='Percentage', color='Algorithm',
                                  barmode='group', text='Percentage', title="Topic Size Distribution: LDA vs NMF",
                                  labels={'Percentage': 'Percentage of Corpus (%)'},
                                  color_discrete_sequence=['#1E88E5', '#FFB300'])
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Show side-by-side keyword profiles
                st.subheader("🔍 Keyword Profile & Semantic Mapping")
                comp_data = []
                for i in range(num_topics):
                    t_id = f"Topic {i+1}"
                    comp_data.append({
                        "Topic": t_id,
                        "LDA Keywords": ", ".join(lda_keywords[t_id][:6]),
                        "LDA Theme Name": lda_names[i],
                        "NMF Keywords": ", ".join(nmf_keywords[t_id][:6]),
                        "NMF Theme Name": nmf_names[i]
                    })
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
            with tab_comp2:
                st.subheader("🌐 LDA pyLDAvis Visualization")
                with st.spinner("Rendering LDA interactive visualization..."):
                    import pyLDAvis
                    vis_lda = get_lda_vis(lda_model, lda_tf, lda_vec)
                    html_lda = pyLDAvis.prepared_data_to_html(vis_lda)
                import streamlit.components.v1 as components
                components.html(html_lda, height=850, width=1300, scrolling=True)
                
            with tab_comp3:
                st.subheader("🌐 NMF pyLDAvis Visualization")
                with st.spinner("Rendering NMF interactive visualization..."):
                    import pyLDAvis
                    # Prepare count matrix for NMF pyLDAvis
                    count_vec = CountVectorizer(max_df=0.95, min_df=2, vocabulary=nmf_vec.vocabulary_)
                    tf_matrix_nmf = count_vec.fit_transform(nmf_prep)
                    vis_nmf = get_nmf_vis(nmf_model, tf_matrix_nmf, count_vec, nmf_dist)
                    html_nmf = pyLDAvis.prepared_data_to_html(vis_nmf)
                import streamlit.components.v1 as components
                components.html(html_nmf, height=850, width=1300, scrolling=True)

    st.markdown("""
        <div style='background-color: #F8F9FA; padding: 1rem; border-radius: 10px; border-left: 5px solid #000000; margin-top: 2rem;'>
            <strong>Quick Guide:</strong> Use the <strong>Interactive pyLDAvis Visualization</strong> to evaluate topic relevance and inter-topic distance.
            The <strong>Topic Distribution</strong> and <strong>Explorer</strong> sections help dive into keyword lists and sample texts.
        </div>
    """, unsafe_allow_html=True)

