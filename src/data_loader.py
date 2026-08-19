from datasets import load_dataset
import pandas as pd

def load_ag_news_data():
    """
    Loads the AG News dataset from Hugging Face Datasets.
    Returns train and test pandas DataFrames.
    """
    print("Loading AG News dataset...")
    # Load dataset - will download if not present
    dataset = load_dataset("ag_news")
    
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Label mapping
    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    
    train_df['label_name'] = train_df['label'].map(label_map)
    test_df['label_name'] = test_df['label'].map(label_map)
    
    print(f"Data Loaded: {len(train_df)} train samples, {len(test_df)} test samples.")
    return train_df, test_df
