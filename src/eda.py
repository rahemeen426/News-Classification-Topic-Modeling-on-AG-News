import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def plot_length_distributions(df):
    """
    Plots box plots for Character Count and Word Count per Category.
    """
    if 'char_count' not in df.columns or 'word_count' not in df.columns or 'label_name' not in df.columns:
        return None, None
        
    # 1. Character Count Distribution
    fig_char = px.box(df, x='label_name', y='char_count', color='label_name',
                      title="Character Count Distribution by Category",
                      points="outliers", # Show outliers
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_char.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Category", yaxis_title="Characters", showlegend=False)
    
    # 2. Word Count Distribution
    fig_word = px.box(df, x='label_name', y='word_count', color='label_name',
                      title="Word Count Distribution by Category",
                      points="outliers",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_word.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Category", yaxis_title="Words", showlegend=False)
    
    return fig_char, fig_word

def get_extreme_articles(df, n=5):
    """
    Returns the top N longest and shortest articles based on word count.
    """
    if 'word_count' not in df.columns:
        return None, None
        
    longest = df.nlargest(n, 'word_count')[['label_name', 'word_count', 'text']]
    shortest = df.nsmallest(n, 'word_count')[['label_name', 'word_count', 'text']]
    
    return longest, shortest

def detect_length_outliers(df, threshold=1.5):
    """
    Detects outliers in word counts using the IQR method.
    Returns a dataframe of outliers.
    """
    if 'word_count' not in df.columns:
        return pd.DataFrame()
        
    Q1 = df['word_count'].quantile(0.25)
    Q3 = df['word_count'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - (threshold * IQR)
    upper_bound = Q3 + (threshold * IQR)
    
    outliers = df[(df['word_count'] < lower_bound) | (df['word_count'] > upper_bound)]
    return outliers[['label_name', 'word_count', 'text']]

def get_statistical_summary(df):
    """
    Calculates comprehensive statistics (Mean, Median, Mode, Variance)
    for word_count and char_count.
    """
    if 'word_count' not in df.columns or 'char_count' not in df.columns:
        return pd.DataFrame()
        
    # Group by Category
    grouped = df.groupby('label_name')[['word_count', 'char_count']]
    
    # Custom Mode function (pandas groupby mode is tricky, taking first mode)
    def mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else 0

    agg_funcs = ['mean', 'median', mode, 'var']
    
    summary = grouped.agg(agg_funcs)
    
    # Flatten Hierarchical Index
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Rename for readability
    rename_map = {
        'word_count_mean': 'Avg Words',
        'word_count_median': 'Median Words',
        'word_count_mode': 'Mode Words',
        'word_count_var': 'Word Var',
        'char_count_mean': 'Avg Chars',
        'char_count_median': 'Median Chars',
        'char_count_mode': 'Mode Chars',
        'char_count_var': 'Char Var'
    }
    summary = summary.rename(columns=rename_map)
    return summary.round(2)

def get_category_proportions(df):
    """Returns category counts and proportions."""
    if 'label_name' not in df.columns:
        return pd.DataFrame()
    # Count per category
    count_df = df['label_name'].value_counts().reset_index()
    count_df.columns = ['Category', 'Count']
    # Proportion per category
    prop_df = df['label_name'].value_counts(normalize=True).reset_index()
    prop_df.columns = ['Category', 'Proportion']
    merged = pd.merge(count_df, prop_df, on='Category')
    merged['Proportion'] = (merged['Proportion'] * 100).round(2).astype(str) + '%'
    return merged



def get_outlier_proportions(df, threshold=1.5):
    """Returns counts and proportions of outliers per category relative to total outliers."""
    if 'word_count' not in df.columns:
        return pd.DataFrame()
    # Get all outliers first
    outliers = detect_length_outliers(df, threshold)
    if outliers.empty:
        return pd.DataFrame(columns=['Category', 'Outlier Count', 'Proportion'])
    total_outliers = len(outliers)
    # Count per category
    counts = outliers['label_name'].value_counts().reset_index()
    counts.columns = ['Category', 'Outlier Count']
    # Proportion of total outliers
    counts['Proportion'] = (counts['Outlier Count'] / total_outliers * 100).round(2).astype(str) + '%'
    return counts
