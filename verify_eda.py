import sys
import os

# Add local directory to path
sys.path.append(os.getcwd())

try:
    from src.eda import plot_length_distributions, get_extreme_articles, detect_length_outliers
    print("SUCCESS: Imported all functions correctly.")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"FAILURE: {e}")
