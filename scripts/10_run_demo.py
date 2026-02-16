"""
Script 10: Launch Streamlit demo application.
Usage: streamlit run scripts/10_run_demo.py
       OR: python scripts/10_run_demo.py (launches streamlit programmatically)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


if __name__ == "__main__":
    demo_path = os.path.join(os.path.dirname(__file__), '..', 'demo', 'app.py')
    os.system(f"streamlit run {demo_path}")
