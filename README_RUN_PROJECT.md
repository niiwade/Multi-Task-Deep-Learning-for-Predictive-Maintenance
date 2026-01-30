# 🚀 HOW TO RUN THE ENTIRE PROJECT - COMPLETE GUIDE

## Project Overview
Multi-task deep learning model for industrial predictive maintenance using Temporal Convolutional Networks (TCN) with attention mechanisms.

**Performs 3 tasks simultaneously:**
- Binary failure prediction (will machine fail?)
- Failure type classification (which failure mode?)
- Time-to-failure regression (when will it fail?)

---

## ⚡ QUICK START (15 MINUTES)

### If You Just Want the Final Report with Visualizations:

```bash
# Step 1: Install dependencies
pip install matplotlib seaborn numpy python-docx

# Step 2: Generate visualizations
python quick_generate.py

# Step 3: Create final report
python generate_report_with_figures.py

# Step 4: Open the report
# Double-click: finalreport_with_figures.docx
```

**Output:** Professional 3-page report with 6 embedded visualizations ✅

---

## 📋 COMPLETE PROJECT EXECUTION (If You Have Trained Models)

### What You Need First:
- ✅ Python 3.6+
- ✅ Trained model checkpoints (if running evaluation)
- ✅ Dataset (AI4I 2020 dataset)

### Complete Workflow:

#### Phase 1: Data Preparation
```bash
python data_preprocessing.py
```
- Loads AI4I 2020 dataset
- Normalizes features
- Handles class imbalance
- Saves processed data

#### Phase 2: Train Base Model (Phase 1)
```bash
python train.py
```
- Trains baseline TCN model
- Performs binary failure detection
- Outputs: `checkpoints/phase1/best_model.pt`
- Time: ~30-60 minutes

#### Phase 3: Phase 2 Training (Enhanced Features)
```bash
python train_phase2.py
```
- Adds temporal features
- Enhanced augmentation
- Better regularization
- Outputs: `checkpoints/phase2/best_model.pt`
- Time: ~30-60 minutes

#### Phase 4: Phase 3 Training (Multi-Task Learning)
```bash
python train_phase3.py
```
- Multi-task learning (failure + type + TTF)
- Improved loss functions
- Better hyperparameters
- Outputs: `checkpoints/phase3/best_model.pt`
- Time: ~60-90 minutes

#### Phase 5: Phase 4 Training (Advanced - Optional)
```bash
python train_phase4.py
```
- ⚠️ WARNING: This phase caused -41% regression
- Not recommended for deployment
- For research/experimentation only
- Time: ~60-90 minutes

#### Phase 6: Evaluation
```bash
python evaluate.py
```
- Evaluates best model on test set
- Generates metrics and confusion matrices
- Outputs: `results.md` with detailed results
- Time: ~5-10 minutes

#### Phase 7: Generate Report with Visualizations
```bash
python quick_generate.py
python generate_report_with_figures.py
```
- Creates professional visualizations
- Generates final DOCX report
- Outputs: `finalreport_with_figures.docx`
- Time: ~5 minutes

---

## 📊 EXPECTED OUTPUTS

### After Data Preprocessing:
```
✓ Normalized features
✓ Train/val/test splits
✓ Class imbalance handled
```

### After Training Phases:
```
checkpoints/
├── phase1/
│   ├── best_model.pt          (F1: 0.5560)
│   └── training_history.json
├── phase2/
│   ├── best_model.pt          (F1: 0.5577)
│   └── training_history.json
├── phase3/
│   ├── best_model.pt          (F1: 0.5545) ← BEST
│   └── training_history.json
└── phase4/
    ├── best_model.pt          (F1: 0.3267) ⚠️
    └── training_history.json
```

### After Evaluation:
```
results.md                      # Detailed evaluation metrics
visualizations/
├── confusion_matrix.png        # Shows 35% miss rate
├── phase_comparison.png        # Shows Phase 4 failure
├── training_curves.png         # Shows convergence
├── failure_types.png           # Shows rare class problem
├── roc_curve.png               # Shows AUC misleading
└── precision_recall_curve.png  # Shows real metric
```

### Final Report:
```
finalreport_with_figures.docx   # Professional 3-page report
✓ Executive Q&A
✓ 6 embedded visualizations
✓ Real Phase 1-4 results
✓ 8 critical limitations
✓ Actionable recommendations
✓ Key lessons learned
```

---

## 🎯 RECOMMENDED WORKFLOWS

### Workflow A: Quick Report Only (15 min)
```bash
pip install matplotlib seaborn numpy python-docx
python quick_generate.py
python generate_report_with_figures.py
# Done! Open finalreport_with_figures.docx
```

### Workflow B: Evaluate Existing Model (30 min)
```bash
pip install -r requirements.txt
python evaluate.py
python quick_generate.py
python generate_report_with_figures.py
```

### Workflow C: Full Training Pipeline (6-8 hours)
```bash
pip install -r requirements.txt
python data_preprocessing.py
python train.py                    # Phase 1
python train_phase2.py             # Phase 2
python train_phase3.py             # Phase 3
python evaluate.py
python quick_generate.py
python generate_report_with_figures.py
```

---

## 📦 INSTALLATION

### Step 1: Install Python
Required: Python 3.6 or higher
```bash
python --version
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install python-docx
pip install imbalanced-learn
```

### Step 3: Download Dataset
The AI4I 2020 dataset should be in `dataset/` folder:
```
dataset/
└── ai4i2020.csv
```

---

## 🔧 KEY SCRIPTS

| Script | Purpose | Input | Output | Time |
|--------|---------|-------|--------|------|
| data_preprocessing.py | Prepare data | ai4i2020.csv | Processed data | 5 min |
| train.py | Phase 1 training | Processed data | best_model.pt | 30-60 min |
| train_phase2.py | Phase 2 training | Phase 1 model | best_model.pt | 30-60 min |
| train_phase3.py | Phase 3 training | Phase 2 model | best_model.pt | 60-90 min |
| train_phase4.py | Phase 4 training | Phase 3 model | best_model.pt | 60-90 min |
| evaluate.py | Evaluate model | best_model.pt | results.md | 5-10 min |
| quick_generate.py | Generate visuals | results | 6 PNG files | 5 min |
| generate_report_with_figures.py | Create report | PNGs + MD | DOCX file | 2 min |

---

## 📂 PROJECT STRUCTURE

```
Project/
├── README_RUN_PROJECT.md           ← YOU ARE HERE
├── requirements.txt                ← Install dependencies
│
├── Data & Config
├── dataset/
│   └── ai4i2020.csv               # Input data
├── config.py                       # Configuration
│
├── Training Scripts
├── train.py                        # Phase 1
├── train_phase2.py                 # Phase 2
├── train_phase3.py                 # Phase 3
├── train_phase4.py                 # Phase 4
│
├── Model & Architecture
├── model.py                        # Base model
├── model_advanced.py               # Advanced architecture
│
├── Evaluation & Analysis
├── evaluate.py                     # Evaluate model
├── evaluate_ensemble.py            # Ensemble evaluation
├── visualize_attention.py          # Attention visualization
│
├── Report Generation
├── quick_generate.py               # Generate visualizations ⭐
├── generate_visualizations.py      # Alt visualization script
├── generate_report_with_figures.py # Create DOCX report
│
├── Outputs (Created)
├── checkpoints/
│   ├── phase1/best_model.pt
│   ├── phase2/best_model.pt
│   ├── phase3/best_model.pt
│   └── phase4/best_model.pt
├── visualizations/                 # PNG files (300 DPI)
├── finalreport.md                  # Markdown report
├── finalreport_with_figures.docx   # Final DOCX report ✅
└── results.md                      # Evaluation results
```

---

## ⚡ QUICK COMMANDS

### Most Important Commands:
```bash
# Just want the report? (15 min)
pip install matplotlib seaborn numpy python-docx && python quick_generate.py && python generate_report_with_figures.py

# Full training? (6-8 hours)
pip install -r requirements.txt && python data_preprocessing.py && python train.py && python train_phase2.py && python train_phase3.py && python evaluate.py && python quick_generate.py && python generate_report_with_figures.py

# Just evaluate? (30 min)
pip install -r requirements.txt && python evaluate.py && python quick_generate.py && python generate_report_with_figures.py
```

---

## 🎯 CONFIGURATION

### Key Settings (in config.py):
```python
# Model parameters
TCN_LAYERS = 4              # Number of dilated conv layers
DILATIONS = [1, 2, 4, 8]   # Dilation factors
DROPOUT = 0.3              # Dropout rate

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5

# Loss functions
USE_TVERSKY_LOSS = True
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3

# Data parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

---

## 📊 EXPECTED RESULTS

### Phase 1 (Baseline):
- F1-Score: 0.5560
- Accuracy: 97.9%
- Precision: 48.3%
- Recall: 65.1%

### Phase 2 (Enhanced):
- F1-Score: 0.5577 (+0.3%)
- Marginal improvement
- Better stability

### Phase 3 (Best):
- F1-Score: 0.5545 (-0.2%)
- Best validation performance
- Multi-task learning benefit

### Phase 4 (Not Recommended):
- F1-Score: 0.3267 (-41%) ❌
- SMOTE + class-balanced batching incompatible
- Causes catastrophic regression

**Best Model:** Phase 3 (F1: 0.5545)

---

## ⚠️ IMPORTANT NOTES

### Phase 4 Warning:
⚠️ **DO NOT USE PHASE 4 FOR DEPLOYMENT**
- SMOTE incompatible with class-balanced batching
- Causes -41% F1 regression
- Use Phase 3 instead

### Deployment Readiness:
❌ **NOT SAFE for autonomous systems**
- 35% failure miss rate (15 false negatives out of 43)
- Only suitable for human-in-the-loop systems

### Performance Limitations:
🔴 **Targets NOT Met:**
- Binary F1: 0.5545 vs target >0.95
- TTF MAE: 3.50 hrs vs target <2.0
- Root cause: Data quality (97:3 imbalance, pseudo-temporal sequences)

---

## 🐛 TROUBLESHOOTING

### Problem: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### Problem: "No such file: ai4i2020.csv"
- Make sure dataset is in `dataset/` folder
- Check filename is exactly `ai4i2020.csv`

### Problem: CUDA out of memory
Edit `config.py`:
```python
BATCH_SIZE = 32  # Reduce from 64
```

### Problem: Training too slow
- Reduce EPOCHS to 10 for testing
- Use smaller BATCH_SIZE
- Run on GPU if available

### Problem: Cannot open DOCX file
- Ensure Word is installed
- Try: Right-click → Open With → Word
- Or use LibreOffice

---

## ✅ VERIFICATION CHECKLIST

Before submitting the report:

- [ ] Python 3.6+ installed
- [ ] All dependencies installed
- [ ] Dataset file exists (ai4i2020.csv)
- [ ] All scripts run without errors
- [ ] Visualizations generated (6 PNG files)
- [ ] DOCX file created successfully
- [ ] Can open DOCX in Word
- [ ] All 6 images visible in report
- [ ] Text is readable and properly formatted

---

## 📈 TIMELINE

| Phase | Task | Time |
|-------|------|------|
| 1 | Install dependencies | 1 min |
| 2 | Data preprocessing | 5 min |
| 3 | Phase 1 training | 30-60 min |
| 4 | Phase 2 training | 30-60 min |
| 5 | Phase 3 training | 60-90 min |
| 6 | Phase 4 training | 60-90 min |
| 7 | Evaluation | 5-10 min |
| 8 | Visualizations | 5 min |
| 9 | Report generation | 2 min |
| **TOTAL** | **Full pipeline** | **6-8 hours** |

---

## 🎯 THREE USAGE SCENARIOS

### Scenario 1: "Just Give Me the Report" (15 min)
```bash
pip install matplotlib seaborn numpy python-docx
python quick_generate.py
python generate_report_with_figures.py
```
✅ Output: finalreport_with_figures.docx

### Scenario 2: "I Have a Trained Model" (30 min)
```bash
pip install -r requirements.txt
python evaluate.py
python quick_generate.py
python generate_report_with_figures.py
```
✅ Output: Results + finalreport_with_figures.docx

### Scenario 3: "Full Training from Scratch" (6-8 hours)
```bash
pip install -r requirements.txt
python data_preprocessing.py
python train.py
python train_phase2.py
python train_phase3.py
python evaluate.py
python quick_generate.py
python generate_report_with_figures.py
```
✅ Output: Trained models + Results + Report

---

## 📞 HELPFUL GUIDES

Want more details? Read these guides:

- **📖_START_HERE.md** - Master index (2 min read)
- **HOW_TO_RUN.md** - Step-by-step guide (5 min read)
- **CODE_SUMMARY.md** - Script descriptions (10 min read)
- **finalreport.md** - Full project report (15 min read)

---

## 🚀 NEXT STEPS

### Option A: Get Report Quickly (Recommended)
1. Run: `pip install matplotlib seaborn numpy python-docx`
2. Run: `python quick_generate.py`
3. Run: `python generate_report_with_figures.py`
4. Open: `finalreport_with_figures.docx` ✅

### Option B: Full Training
1. Run: `pip install -r requirements.txt`
2. Run: `python data_preprocessing.py`
3. Run: `python train.py` (+ phase 2, 3 if desired)
4. Run: `python evaluate.py`
5. Run: `python quick_generate.py && python generate_report_with_figures.py`
6. Open: `finalreport_with_figures.docx` ✅

---

## 💡 KEY TAKEAWAYS

✅ **What Worked:**
- TCN architecture suitable for sequential data
- Multi-task learning improves generalization
- Attention mechanisms add interpretability
- Proper regularization prevents overfitting

❌ **What Didn't:**
- Class imbalance (97:3) too extreme
- Pseudo-temporal sequences not real time-series
- Targets unrealistic for dataset size (43 test failures)
- SMOTE + class-balanced batching incompatible

🎯 **Key Insight:**
**Data quality is the limiting factor, not model design.**

---

## 📄 FINAL DELIVERABLE

**Main Output:** `finalreport_with_figures.docx`

**Contains:**
- ✅ 3-page comprehensive report
- ✅ 6 professional visualizations
- ✅ Real Phase 1-4 results
- ✅ 8 critical limitations
- ✅ Actionable recommendations
- ✅ Key lessons learned
- ✅ Professional formatting

**Ready for submission!** 🎉

---

## ⏱️ TIME TO COMPLETION

- **Just want report?** → 15 minutes ⚡
- **Evaluate existing model?** → 30 minutes
- **Full training pipeline?** → 6-8 hours 🔄

---

**Start now! Choose your workflow above and get going.** 🚀

For questions, refer to the detailed guides in the project folder.
