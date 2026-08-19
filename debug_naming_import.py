import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

try:
    from src.topic_model import suggest_topic_name
    print("SUCCESS: suggest_topic_name imported correctly.")
    test_kws = ['stock', 'market', 'nasdaq']
    print(f"Test Call: {suggest_topic_name(test_kws)}")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
