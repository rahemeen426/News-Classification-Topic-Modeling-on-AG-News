import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, 
                    text_auto=True, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels,
                    y=labels,
                    title="Confusion Matrix")
    return fig

def plot_topic_distribution(lda, tf, n_topics):
    # This is a bit complex to visualize simply without pyLDAvis. 
    # For now, we can plot the distribution of topics across the dataset? 
    # Or just top words.
    pass

def plot_top_words(topics_dict):
    # Visualize top words for each topic as horizontal bar charts or similar
    # For simplicity, we might just show them as text or tables in Streamlit, 
    # or create a bar chart for one topic at a time.
    pass
