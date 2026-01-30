@echo off
REM Quick Start Guide: Generate Report with Visualizations (Windows)
REM Run: generate_all.bat

echo ==================================================
echo QUICK START: GENERATE REPORT WITH VISUALIZATIONS
echo ==================================================

echo.
echo [1/4] Checking Python installation...
python --version

echo.
echo [2/4] Installing dependencies...
pip install matplotlib seaborn numpy python-docx --quiet

echo.
echo [3/4] Generating 6 visualizations...
python generate_visualizations.py

echo.
echo [4/4] Creating final report with embedded figures...
python generate_report_with_figures.py

echo.
echo ==================================================
echo OK COMPLETE!
echo ==================================================
echo.
echo Generated Files:
echo    - visualizations\confusion_matrix.png
echo    - visualizations\phase_comparison.png
echo    - visualizations\training_curves.png
echo    - visualizations\failure_types.png
echo    - visualizations\roc_curve.png
echo    - visualizations\precision_recall_curve.png
echo    - finalreport_with_figures.docx
echo.
echo Next Steps:
echo    1. Open finalreport_with_figures.docx in Word
echo    2. Review all visualizations and captions
echo    3. Export as PDF for submission
echo.
pause
