"""
Test Runner for TikTok Review Analysis Dashboard
Run all tests to validate the system functionality
"""

import subprocess
import sys
import os
from pathlib import Path

def run_test(test_file):
    """Run a single test file"""
    try:
        print(f"🧪 Running {test_file}...")
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd='tests')
        
        if result.returncode == 0:
            print(f"✅ {test_file} passed")
            return True
        else:
            print(f"❌ {test_file} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"💥 Error running {test_file}: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TikTok Review Analysis - Test Suite")
    print("=" * 50)
    
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ Tests directory not found!")
        return
    
    # Get all test files
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("⚠️ No test files found!")
        return
    
    print(f"Found {len(test_files)} test files\n")
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if run_test(test_file.name):
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main()
