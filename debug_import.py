import sys
import os

# Ensure cwd is in path
sys.path.append(os.getcwd())

try:
    from src.models import load_transformer_model
    print("SUCCESS: Imported load_transformer_model")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
