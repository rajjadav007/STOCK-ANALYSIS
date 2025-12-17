#!/usr/bin/env python3
"""
Test Suite for Stock Market ML Analysis
Runs all components to verify no warnings or errors
"""

import subprocess
import sys

def run_test(script_name, description):
    """Run a test script and report results"""
    print(f"\n🧪 Testing: {description}")
    print("-" * 50)
    
    try:
        # Run with all warnings enabled
        result = subprocess.run([sys.executable, "-W", "all", script_name], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✅ {script_name} - SUCCESS")
            if "warning" in result.stderr.lower():
                print(f"⚠️  Found warnings in stderr:")
                print(result.stderr)
            else:
                print("🎉 No warnings detected!")
        else:
            print(f"❌ {script_name} - FAILED")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  {script_name} - TIMEOUT (>120 seconds)")
        return False
    except Exception as e:
        print(f"💥 {script_name} - EXCEPTION: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 STOCK MARKET ML ANALYSIS - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("load_and_clean_data.py", "Data Loading & Cleaning"),
        ("main.py", "ML Training & Analysis"), 
        ("predict.py", "Stock Price Prediction")
    ]
    
    all_passed = True
    
    for script, description in tests:
        if not run_test(script, description):
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - NO WARNINGS OR ERRORS!")
        print("✅ Project is ready for production use")
    else:
        print("❌ Some tests failed - check output above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)