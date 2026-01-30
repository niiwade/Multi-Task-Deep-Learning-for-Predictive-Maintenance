#!/bin/bash
# Quick Start Guide: Generate Report with Visualizations
# Usage: Run this script or follow the steps below manually

echo "=================================================="
echo "QUICK START: GENERATE REPORT WITH VISUALIZATIONS"
echo "=================================================="

# Step 1: Check Python installation
echo ""
echo "[1/3] Checking Python installation..."
python --version

# Step 2: Install dependencies
echo ""
echo "[2/3] Installing dependencies..."
pip install matplotlib seaborn numpy python-docx --quiet

# Step 3: Generate visualizations
echo ""
echo "[3/3] Generating visualizations..."
python generate_visualizations.py

# Step 4: Create DOCX with figures
echo ""
echo "[4/4] Creating final report with embedded figures..."
python generate_report_with_figures.py

echo ""
echo "=================================================="
echo "✅ COMPLETE!"
echo "=================================================="
echo ""
echo "📊 Generated Files:"
echo "   • visualizations/confusion_matrix.png"
echo "   • visualizations/phase_comparison.png"
echo "   • visualizations/training_curves.png"
echo "   • visualizations/failure_types.png"
echo "   • visualizations/roc_curve.png"
echo "   • visualizations/precision_recall_curve.png"
echo "   • finalreport_with_figures.docx"
echo ""
echo "📋 Next Steps:"
echo "   1. Open finalreport_with_figures.docx in Word"
echo "   2. Review all visualizations and captions"
echo "   3. Export as PDF for submission"
echo ""
