# Machine Learning Classification of Plasma Confinement Mode in the DIII-D Tokamak

A machine learning approach to classify H-mode vs L-mode confinement regimes in tokamak plasmas, achieving **98.7% accuracy** on expert-labelled DIII-D data. This was carried out as part of UCL's Faculty of Computer Science MSc Data Science and Machine Learning thesis.

## 🔬 Project Overview

This repository contains the final implementation of an enhanced machine learning model for automated H-mode classification in fusion plasmas. The model significantly outperforms existing methods, namely the OMFIT H-mode studies module (91.1% accuracy), by using physics-informed feature engineering and advanced gradient-based analysis.

### Key Achievements
- **98.7% test accuracy** on DIII-D tokamak data
- **25 physics-based features** capturing pedestal structure and profile gradients
- **Gradient Boosting Classifier** with optimized hyperparameters
- **Comprehensive uncertainty quantification** and validation

## 📊 Model Performance

| Method | Accuracy | Features | Data Processing |
|--------|----------|----------|----------------|
| **This Work** | **98.71%** | 25 enhanced physics features | Raw profile analysis |
| OMFIT Baseline | 91.1% | Basic gradient features | Pre-processed profiles |

## 🚀 Quick Start

### ⚠️ IMPORTANT: Required Files

To achieve the advertised **98-99% accuracy**, you need **BOTH** of these files:

1. ✅ `OMFIT_Hmode_Studies.pkl` (~423 MB) - Plasma profile data
2. ✅ `Labels.xlsx` (~26 KB) - **CRITICAL**: Hand-labeled ground truth data

**Without `Labels.xlsx`, the model will only achieve ~90% accuracy** (using automated Hflag labels instead of ground truth).

### Step 1: Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv hmode_env

# Activate the environment
source hmode_env/bin/activate  # On Windows: hmode_env\Scripts\activate

# Install required packages
pip install numpy scipy scikit-learn matplotlib pandas openpyxl
```

### Step 2: Verify Data Files

```bash
# Ensure both files are present in the project directory
ls -lh OMFIT_Hmode_Studies.pkl Labels.xlsx

# You should see:
# -rw-r--r--  423M  OMFIT_Hmode_Studies.pkl
# -rw-r--r--   26K  Labels.xlsx
```

### Step 3: Run the Classifier

```bash
python hmode_classifier.py
```

**Expected output:**
```
✓ Data loaded: 251 shots available
✓ Ground truth labels loaded: 749 entries
✓ Dataset built: ~8200 samples, 24 features
✓ Using ground truth labels from Labels.xlsx
✓ Test accuracy: 0.9909 (99.09%)
```

### Alternative: Python API Usage

```python
from hmode_classifier import HModeClassifier

# Initialize with both data files
classifier = HModeClassifier(
    data_file='OMFIT_Hmode_Studies.pkl',
    labels_file='Labels.xlsx'  # CRITICAL for 98%+ accuracy
)

# Load data
classifier.load_data()

# Build dataset (uses ground truth labels)
dataset = classifier.build_dataset()

# Train the final optimized model
results = classifier.train_final_model(dataset['features'], dataset['labels'])

# Make predictions
prediction = classifier.predict(classifier.DD_full, shot_id=202982, t_index=0)
print(f"Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.4f})")
```

### Troubleshooting

**If you see this warning:**
```
⚠ Warning: Ground truth labels file 'Labels.xlsx' not found.
  The model will use Hflag from the pickle file, but accuracy may be lower.
```

**Action required:** Obtain the `Labels.xlsx` file - it contains expert hand-labeled data that is essential for achieving 98%+ accuracy.

**If accuracy is only ~85-90%:**
- Check that `Labels.xlsx` is in the same directory as the script
- Verify the output shows: `✓ Ground truth labels loaded: 749 entries`
- Expected sample count: ~8,200 (not ~1,200)

## 🧠 Model Architecture

### Feature Engineering Pipeline

The model uses 25 physics-informed features organized into 6 categories:

1. **Core Gradient Features** (3 features)
   - Maximum gradients of temperature, density, and pressure in pedestal region

2. **Pedestal Structure** (6 features)  
   - Height differences and edge values for all profiles

3. **Profile Shape** (6 features)
   - Steepness metrics and peaking factors

4. **Curvature Features** (3 features)
   - Second derivatives capturing profile curvature

5. **Multi-field Coupling** (3 features)
   - Gradient correlations and combined gradient magnitudes

6. **OMFIT Comparison** (3 features)
   - Benchmarking against existing processed data

### Machine Learning Model

**Gradient Boosting Classifier** with optimized hyperparameters:
```python
GradientBoostingClassifier(
    n_estimators=150,        # Optimal from validation curves
    max_depth=3,             # Prevents overfitting  
    learning_rate=0.1,       # Good convergence
    subsample=0.8,           # Stochastic boosting
    max_features='sqrt',     # Feature sampling
    validation_fraction=0.1,  # Early stopping
    n_iter_no_change=10,     # Patience
    random_state=42
)
```

## 🔬 Scientific Methodology

### Physics-Based Feature Design

The feature engineering is grounded in tokamak physics:

- **Pedestal Region**: ψ_N ∈ [0.85, 0.98] captures the key H-mode pedestal
- **Profile Analysis**: Temperature, density, and pressure profiles from Thomson scattering
- **Gradient Calculations**: First and second derivatives reveal transport barriers
- **Multi-field Coupling**: Cross-correlations between different plasma parameters

### Data Processing Pipeline

1. **Raw Profile Loading**: Direct access to Thomson scattering measurements
2. **Interpolation & Smoothing**: Spline interpolation with median filtering
3. **Gradient Computation**: Numerical derivatives with uncertainty handling
4. **Feature Extraction**: Physics-motivated feature calculations
5. **Standardization**: Feature scaling for ML compatibility

### Validation Strategy

- **Stratified train/test split**: 80/20 with class balance preservation
- **5-fold cross-validation**: Robust performance estimation
- **Overfitting analysis**: Validation curves and early stopping
- **Bootstrap stability**: Model consistency across different data samples

## 📈 Key Results

### Classification Performance

**With Ground Truth Labels (Labels.xlsx):**
```
                precision    recall  f1-score   support
    L-mode         0.99      0.99      0.99       599
    H-mode         0.99      0.99      0.99      1042

    accuracy                           0.99      1641
   macro avg       0.99      0.99      0.99      1641
weighted avg       0.99      0.99      0.99      1641
```

**Key Metrics:**
- Test Accuracy: **99.09%**
- Cross-Validation: **98.87% ± 0.30%**
- Total Samples: **8,201**
- L-mode: 2,996 samples
- H-mode: 5,205 samples

### Feature Importance

Top 5 most important features:
1. **omfit_max_grad_pe** (0.2394) - Maximum pressure gradient (OMFIT processed)
2. **pe_peaking** (0.1781) - Pressure peaking factor
3. **pe_edge** (0.1373) - Edge pressure value
4. **pe_steepness** (0.1240) - Pressure profile steepness
5. **omfit_max_grad_ne** (0.1184) - Maximum density gradient (OMFIT processed)

## 🏗️ Code Structure

```
hmode_classifier.py
├── HModeClassifier                    # Main class
│   ├── __init__()                     # Initialize with data files
│   ├── load_data()                    # Load OMFIT dataset + Labels.xlsx
│   ├── get_ground_truth_label()       # Get labels from Excel
│   ├── interpolate_scaled_profile()   # Profile processing
│   ├── compute_enhanced_features()    # Feature engineering
│   ├── build_dataset()                # Complete dataset construction
│   ├── train_final_model()            # Model training & validation
│   ├── predict()                      # Single shot prediction
│   ├── plot_feature_importance()      # Visualization
│   ├── save_model()                   # Save trained model
│   └── load_model()                   # Load trained model
└── main()                             # Complete pipeline demo
```

### Project Files

- `hmode_classifier.py` - Main implementation
- `OMFIT_Hmode_Studies.pkl` - Plasma profile data (**Required**)
- `Labels.xlsx` - Ground truth labels (**Required for 98%+ accuracy**)
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `SUPERVISOR_README.md` - Detailed setup guide for reproduction

## 🔬 Physics Background

### H-mode Confinement

H-mode (High confinement mode) is a crucial plasma regime for fusion energy:

- **Transport Barrier**: Steep pressure gradients at plasma edge
- **Improved Confinement**: ~2x better energy confinement than L-mode
- **ITER Baseline**: Required for economic fusion power plants

### Classification Challenge

Traditional classification methods struggle with:
- **Complex Profile Shapes**: Non-linear pedestal structures
- **Multi-field Coupling**: Temperature, density, pressure interdependencies
- **Temporal Evolution**: Dynamic transitions between modes
- **Measurement Noise**: Diagnostic uncertainties and artifacts

### Why Ground Truth Labels Matter

This implementation achieves 98-99% accuracy by using:

1. **Expert Hand-Labeled Data** (`Labels.xlsx`):
   - 749 time windows manually labeled by plasma physicists
   - Filters out ambiguous periods (D-mode/dithering)
   - Provides clean L-mode vs H-mode classification

2. **Physics-Based Features**:
   - 24 features extracted from raw Thomson scattering profiles
   - Captures pedestal structure, gradients, and multi-field coupling

The automated OMFIT Hflag achieves only ~91% accuracy, which is what this work aimed to surpass.

## 🚧 Future Work

### Immediate Extensions
- Integration of measurement uncertainties (`Rawtemp_e`, `Rawdensity_e`, `Rawpress_e`)
- Adaptive pedestal boundary detection
- Transient mode classification (dithering, ELM-free phases)

### Advanced Developments
- **Physics-Informed Neural Networks**: Incorporating MHD stability constraints
- **Bayesian Uncertainty Quantification**: Model confidence estimation
- **Real-time Implementation**: <10ms prediction latency for plasma control
- **Cross-machine Validation**: Transfer learning to ITER, JET, ASDEX-U

### Long-term Vision  
- **Foundation Models**: Multi-machine pre-training for fusion physics
- **Causal Machine Learning**: Understanding H-mode trigger mechanisms
- **Integrated Control**: Real-time plasma scenario optimization

## 📚 References

1. **ITER Physics Basis** - Progress in the ITER Physics Basis, Nuclear Fusion (2007)
2. **H-mode Discovery** - Wagner et al., Physical Review Letters (1982)
3. **DIII-D Thomson Scattering** - Eldon et al., Review of Scientific Instruments (2018)
4. **Machine Learning in Fusion** - Kates-Harbeck et al., Nature (2019)

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Feature engineering improvements
- Alternative ML architectures  
- Cross-machine validation
- Real-time implementation
- Physics-informed constraints

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **DIII-D National Fusion Facility** for providing the experimental data and being incredibly attentive and helpful throughout
- **University College London** for their regular supervision and support

---

**Citation**: If you use this code in your research, please cite:
```
@misc{hmode_classifier_2024,
  title={Machine Learning Classification of Plasma
Confinement Mode in the DIII-D Tokamak},
  author={Despina Demetriadou},
  year={2024},
  url={https://github.com/despinademetriadou/hmode-classifier}
}
```
