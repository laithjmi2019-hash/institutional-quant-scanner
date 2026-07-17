import sys
import os

# Add root to path so all engine imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Execute the main application
exec(open(os.path.join(os.path.dirname(__file__), "app", "main.py")).read())
