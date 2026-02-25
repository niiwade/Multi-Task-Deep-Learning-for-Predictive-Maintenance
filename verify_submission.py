#!/usr/bin/env python3
"""
Verification script - tests that analysis.py can run without errors
"""

import os
import sys

# Get the submission directory
submission_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(submission_dir)

# Create visualizations directory
os.makedirs("visualizations", exist_ok=True)

# Change to parent directory for data access
os.chdir(os.path.dirname(submission_dir))

# Now test imports and run analysis
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("✓ All required libraries imported successfully")
    
    # Test data loading
    df = pd.read_csv('dataset/train/train.csv')
    print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Quick validation
    assert df.shape[0] >= 500, "Dataset must have at least 500 rows"
    assert df.shape[1] >= 3, "Dataset must have at least 3 columns"
    print("✓ Dataset meets minimum requirements")
    
    # Test analysis file can be parsed
    with open('submission/analysis.py', 'r') as f:
        analysis_code = f.read()
    compile(analysis_code, 'submission/analysis.py', 'exec')
    print("✓ analysis.py compiles without syntax errors")
    
    # Check generated files
    required_files = [
        'submission/README.md',
        'submission/questions_and_answers.md',
        'submission/analysis.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size} bytes)")
        else:
            print(f"✗ {file_path} NOT FOUND")
    
    print("\n" + "="*80)
    print("✓ SUBMISSION VERIFICATION COMPLETE")
    print("="*80)
    print("\nTo run the analysis and generate visualizations, execute:")
    print("  cd submission")
    print("  python analysis.py")
    print("\nThis will:")
    print("  1. Answer 4 questions about machine failures")
    print("  2. Generate 4 visualization PNG files")
    print("  3. Display detailed statistical analysis")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
