"""
Wildlife Conservation Tools - Main Entry Point

This module provides the main entry point for the wildlife conservation tools package.
"""

import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the wildlife conservation tools."""
    print("🦁 Wildlife Conservation Tools")
    print("=" * 50)
    print("AI-powered threat prediction and conservation planning")
    print()
    print("Available commands:")
    print("  python -m src.data.generate_data     - Generate synthetic data")
    print("  python scripts/train_models.py      - Train and evaluate models")
    print("  streamlit run demo/app.py           - Launch interactive demo")
    print()
    print("For more information, see README.md")
    print("Author: kryptologyst - https://github.com/kryptologyst")


if __name__ == "__main__":
    main()
