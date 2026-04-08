#!/usr/bin/env python3
"""
Quick Start Script for Wildlife Conservation Tools

This script provides a quick way to get started with the wildlife conservation tools.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print("Error:", e.stderr.strip())
        return False

def main():
    """Main quick start function."""
    print("🦁 Wildlife Conservation Tools - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("demo").exists():
        print("❌ Please run this script from the project root directory")
        print("   (the directory containing src/ and demo/ folders)")
        return 1
    
    # Test installation
    print("\n1. Testing installation...")
    if not run_command("python3 test_installation.py", "Installation test"):
        print("❌ Installation test failed. Please check dependencies.")
        return 1
    
    # Generate sample data
    print("\n2. Generating sample data...")
    if not run_command("python3 -m src.data.generate_data", "Data generation"):
        print("❌ Data generation failed.")
        return 1
    
    # Train models
    print("\n3. Training models...")
    if not run_command("python3 scripts/train_models.py --n-samples 200 --n-regions 20", "Model training"):
        print("❌ Model training failed.")
        return 1
    
    print("\n🎉 Quick start completed successfully!")
    print("\nNext steps:")
    print("  📊 Launch interactive demo:")
    print("     streamlit run demo/app.py")
    print("\n  📈 View generated assets:")
    print("     ls -la assets/")
    print("\n  📚 Read documentation:")
    print("     cat README.md")
    print("\n  ⚠️  Important:")
    print("     cat DISCLAIMER.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
