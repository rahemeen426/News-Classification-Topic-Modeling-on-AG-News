import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords and lemmatizer safely
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = set()

LEMMATIZER = WordNetLemmatizer()
RE_CLEAN = re.compile(r'[^a-zA-Z\s]')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase & Clean non-alphabetic characters
    cleaned = RE_CLEAN.sub('', text.lower())
    # Fast split tokenization
    words = cleaned.split()
    # Filter stopwords and short words, then lemmatize
    processed = [LEMMATIZER.lemmatize(w) for w in words if w not in STOPWORDS and len(w) > 2]
    return " ".join(processed)

def fit_lda_model(texts, n_topics=5, max_features=2000):
    """
    Fits an LDA model on the provided text list (after preprocessing).
    Returns the LDA model, Count vectorizer, transformed features, and preprocessed texts.
    """
    print(f"Preprocessing {len(texts)} texts for LDA...")
    preprocessed_texts = [preprocess_text(t) for t in texts]
    
    print(f"Fitting LDA Topic Model with {n_topics} topics...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=max_features)
    tf = tf_vectorizer.fit_transform(preprocessed_texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online')
    lda.fit(tf)
    print("LDA Model Fitted.")
    return lda, tf_vectorizer, tf, preprocessed_texts

def fit_nmf_model(texts, n_topics=5, max_features=2000):
    """
    Fits an NMF model on the provided text list (after preprocessing).
    Returns the NMF model, TF-IDF vectorizer, transformed features, and preprocessed texts.
    """
    print(f"Preprocessing {len(texts)} texts for NMF...")
    preprocessed_texts = [preprocess_text(t) for t in texts]
    
    print(f"Fitting NMF Topic Model with {n_topics} topics...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=max_features)
    tfidf = tfidf_vectorizer.fit_transform(preprocessed_texts)
    
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvd')
    nmf.fit(tfidf)
    print("NMF Model Fitted.")
    return nmf, tfidf_vectorizer, tfidf, preprocessed_texts

def get_lda_vis(lda_model, tf_matrix, vectorizer):
    """
    Prepares pyLDAvis data for LDA model.
    """
    import pyLDAvis
    import pyLDAvis.lda_model
    return pyLDAvis.lda_model.prepare(lda_model, tf_matrix, vectorizer)

def get_nmf_vis(nmf_model, tf_matrix, count_vectorizer, doc_topic_dist):
    """
    Prepares pyLDAvis data for NMF model.
    tf_matrix and count_vectorizer should be fitted on the same preprocessed corpus.
    """
    import pyLDAvis
    
    # Normalize topic-term distribution (each topic must sum to 1)
    components = nmf_model.components_
    topic_term_dists = components / components.sum(axis=1, keepdims=True)
    
    # Document-topic distributions (already normalized)
    doc_topic_dists = doc_topic_dist
    
    # Document lengths
    doc_lengths = tf_matrix.sum(axis=1).getA1()
    doc_lengths[doc_lengths == 0] = 1 # Avoid division by zero
    
    # Vocabulary list
    vocab = count_vectorizer.get_feature_names_out().tolist()
    
    # Term frequencies
    term_frequency = tf_matrix.sum(axis=0).getA1()
    
    # Prepare using generic pyLDAvis.prepare
    return pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency,
        sort_topics=False
    )

def get_top_words(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx+1}"] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return topics

def get_document_topics(model, features):
    """
    Returns the document-topic distribution matrix (normalized row-wise).
    """
    dist = model.transform(features)
    row_sums = dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return dist / row_sums

def get_dominant_topics(doc_topic_dist):
    """
    Returns the dominant topic index for each document.
    """
    return np.argmax(doc_topic_dist, axis=1)

def get_topic_stats(dominant_topics, n_topics):
    """
    Returns counts and percentages for each topic.
    """
    counts = np.bincount(dominant_topics, minlength=n_topics)
    total = len(dominant_topics)
    
    df = pd.DataFrame({
        'Topic': [f"Topic {i+1}" for i in range(n_topics)],
        'Count': counts,
        'Percentage': (counts / total * 100).round(2)
    })
    return df

def get_topic_samples(df, dominant_topics, n_samples=3):
    """
    Gets sample articles for each topic.
    """
    samples = {}
    unique_topics = np.unique(dominant_topics)
    
    for t_idx in unique_topics:
        topic_name = f"Topic {t_idx+1}"
        # Filter texts belonging to this topic
        topic_texts = df.iloc[dominant_topics == t_idx]['text'].head(n_samples).tolist()
        samples[topic_name] = topic_texts
    return samples


def get_top_topic_words(model, feature_names, n_top_words=10):
    """Return topic top words for each topic."""
    return [
        [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for topic in model.components_
    ]


def compute_topic_diversity(model, feature_names, n_top_words=10):
    """Compute topic diversity as unique top words fraction."""
    all_top_words = [word for topic in get_top_topic_words(model, feature_names, n_top_words) for word in topic]
    unique_words = set(all_top_words)
    total_words = n_top_words * len(model.components_)
    return len(unique_words) / total_words if total_words > 0 else 0.0


def compute_topic_purity(dominant_topics, true_labels):
    """
    Compute topic purity: fraction of documents where the dominant topic matches the best majority label.
    """
    labels_by_topic = {}
    for topic_idx, label in zip(dominant_topics, true_labels):
        labels_by_topic.setdefault(topic_idx, []).append(label)

    topic_majority = {
        topic_idx: max(set(labels), key=labels.count)
        for topic_idx, labels in labels_by_topic.items()
    }

    matches = sum(1 for topic_idx, label in zip(dominant_topics, true_labels)
                  if topic_majority.get(topic_idx) == label)
    return matches / len(true_labels) if len(true_labels) > 0 else 0.0


def compute_topic_similarity(lda_model):
    """
    Computes cosine similarity between topic word distributions.
    """
    # Components are pseudo-counts for keywords (vectors across vocabulary)
    components = lda_model.components_
    # Normalize to get probability distribution over words
    topic_word_dist = components / components.sum(axis=1)[:, np.newaxis]
    
    # Compute similarity (dot product of normalized vectors)
    similarity = np.dot(topic_word_dist, topic_word_dist.T)
    return similarity

def plot_topic_similarity(similarity_matrix):
    """
    Plots a heatmap of topic similarity.
    """
    n = similarity_matrix.shape[0]
    labels = [f"T{i+1}" for i in range(n)]
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        zmin=0, zmax=1
    ))
    fig.update_layout(title="Topic Similarity Matrix", width=600, height=600)
    return fig

def suggest_topic_name(keywords):
    """
    Heuristic-based topic namer for AG News dataset.
    Maps keyword clusters to human-friendly titles.
    """
    kws = [k.lower() for k in keywords]
    
    # Mapping of anchor words to semantic titles
    category_map = {
        # Business / Economics
        'Stock Market & Finance': ['stock', 'shares', 'market', 'nasdaq', 'dow', 'investor', 'bank', 'earnings', 'profit', 'economy'],
        'Oil & Energy Prices': ['oil', 'crude', 'gas', 'prices', 'barrel', 'energy', 'fuel', 'production'],
        'Retail & Consumer Goods': ['retail', 'sales', 'consumer', 'wal-mart', 'stores', 'shopping'],
        'Corporate Mergers': ['merger', 'acquisition', 'buyout', 'bid', 'takeover'],
        
        # Sci/Tech
        'Computing & Hardware': ['intel', 'processor', 'chip', 'computer', 'hardware', 'laptop', 'desktop', 'pc'],
        'Software & Operating Systems': ['microsoft', 'windows', 'software', 'update', 'security', 'vulnerability', 'linux', 'open-source'],
        'Internet & Web Services': ['google', 'internet', 'search', 'online', 'yahoo', 'web', 'browser', 'service'],
        'Consumer Electronics': ['apple', 'iphone', 'ipod', 'mobile', 'phone', 'cell', 'nokia', 'sony', 'gadget'],
        'Space & Science': ['nasa', 'space', 'shuttle', 'mars', 'orbit', 'planet', 'astronomy', 'discovery', 'science'],
        'Video Games & Entertainment': ['game', 'nintendo', 'sony', 'xbox', 'playstation', 'gaming', 'video'],
        
        # Sports
        'European Soccer': ['football', 'soccer', 'cup', 'league', 'united', 'arsenal', 'chelsea', 'liverpool', 'real', 'madrid', 'uefa', 'fifa'],
        'Basketball (NBA)': ['nba', 'basketball', 'lakers', 'pistons', 'shaq', 'kobe', 'lebron', 'playoffs'],
        'Baseball (MLB)': ['baseball', 'yankees', 'red', 'sox', 'mlb', 'series', 'pitcher', 'homerun'],
        'American Football (NFL)': ['nfl', 'football', 'quarterback', 'super', 'bowl', 'touchdown', 'patriots'],
        'Tennis & Tournaments': ['tennis', 'federer', 'safin', 'williams', 'open', 'wimbledon'],
        'Cycling & Racing': ['cycling', 'armstrong', 'tour', 'france', 'racing', 'formula', 'schumacher'],
        'Olympics & Global Sports': ['olympics', 'athens', 'medal', 'gold', 'silver', 'bronze'],
        
        # World
        'Middle East Conflict': ['iraq', 'israel', 'palestinian', 'baghdad', 'gaza', 'israeli', 'sharon', 'arafat', 'militant'],
        'US Politics & Elections': ['bush', 'kerry', 'election', 'democratic', 'republican', 'senate', 'house', 'president', 'campaign'],
        'International Relations': ['un', 'united', 'nations', 'china', 'russia', 'talks', 'nuclear', 'treaty', 'sanctions'],
        'Military & Security': ['army', 'force', 'troops', 'combat', 'soldiers', 'pentagon', 'defense', 'attack'],
        'Crime & Justice': ['court', 'judge', 'trial', 'justice', 'police', 'prison', 'arrest', 'charges'],
        'Disasters & Public Safety': ['earthquake', 'flood', 'storm', 'hurricane', 'safety', 'emergency', 'death', 'toll']
    }
    
    # Simple scoring: which category has most keyword hits?
    scores = {}
    for title, anchors in category_map.items():
        # Count how many of the top 5 keywords match the anchor list
        match_count = sum(1 for k in kws[:5] if k in anchors)
        if match_count > 0:
            scores[title] = match_count
            
    if scores:
        # Return title with highest score
        best_match = max(scores, key=scores.get)
        return best_match
        
    # Fallback: Just format the top 3 keywords nicely
    return "Theme: " + " | ".join(keywords[:3]).title()
