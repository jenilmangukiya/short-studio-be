#!/usr/bin/env python3
"""Convenience script to start the Shorts Studio queue worker.
Usage:
    python scripts/run_worker.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.worker import main

if __name__ == "__main__":
    main()
