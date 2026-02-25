import os
import sys

# Add parent directory to path to access dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change to submission directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Now run the analysis
exec(open('analysis.py').read())
