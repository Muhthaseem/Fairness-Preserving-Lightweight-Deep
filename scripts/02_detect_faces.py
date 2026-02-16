"""
Script 02: Detect and crop faces from extracted frames.
Usage: python scripts/02_detect_faces.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.face_detector import process_all_faces


if __name__ == "__main__":
    process_all_faces()
