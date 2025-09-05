# Fusion Plasma Confinement State Classification

A machine learning approach to classify H-mode vs L-mode confinement regimes, achieving **98.71% accuracy** on DIII-D data. Please contact demetriadoudespina@gmail.com for provision of the OMFIT_Hmode_Studies.pkl file for the raw data. 

## Project Overview

This repository contains the final implementation of an enhanced machine learning model for automated H-mode classification in fusion plasmas. The model significantly outperforms the existing OMFIT H-mode\_studies preprocessing tool (91.1% accuracy) by using physics-informed feature engineering and advanced gradient-based analysis.

### Key Achievements
- **98.71% test accuracy** on DIII-D tokamak data
- **25 physics-based features** capturing pedestal structure and profile gradients
- **Gradient Boosting Classifier** with optimized hyperparameters
- **Comprehensive uncertainty quantification** and validation

## Model Performance

| Method | Accuracy | Features | Data Processing |
|--------|----------|----------|----------------|
| **This Work** | **98.71%** | 25 enhanced physics features | Raw profile analysis |
| OMFIT Baseline | 91.1% | Basic gradient features | Pre-processed profiles |

## Quick Start

### Prerequisites

```bash
pip install numpy scipy scikit-learn matplotlib
```

### Data Requirements

You need the DIII-D dataset file:
- `OMFIT_Hmode_Studies.pkl` - Contains raw and processed plasma profile data

### Usage

```python
from hmode_classifier import HModeClassifier

# Initialize and train the complete pipeline
classifier = HModeClassifier('OMFIT_Hmode_Studies.pkl')
classifier.load_data()

# Build enhanced feature dataset
dataset = classifier.build_dataset()

# Train the final optimized model
results = classifier.train_final_model(dataset['features'], dataset['labels'])

# Make predictions
prediction = classifier.predict(classifier.DD_full, shot_id=202982, t_index=0)
print(f"Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.4f})")
```

### Command Line Usage

```bash
python hmode_classifier.py
```

This runs the complete pipeline: data loading → feature engineering → model training → evaluation → model saving.

## Model Architecture

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

## Scientific Methodology

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

## Key Results

### Classification Performance
```
                precision    recall  f1-score   support
    L-mode         0.99      0.98      0.99      1234
    H-mode         0.98      0.99      0.98      1189
    
    accuracy                           0.99      2423
   macro avg       0.99      0.99      0.99      2423
weighted avg       0.99      0.99      0.99      2423
```

### Feature Importance

Top 5 most important features:
1. **max_grad_pe** (0.2341) - Maximum pressure gradient in pedestal
2. **max_grad_Te** (0.1876) - Maximum temperature gradient  
3. **pe_steepness** (0.1234) - Pressure profile steepness
4. **max_combined_grad** (0.0987) - Combined gradient magnitude
5. **Te_height** (0.0823) - Temperature pedestal height

## Code Structure

```
hmode_classifier.py
├── HModeClassifier          # Main class
│   ├── load_data()          # Load OMFIT dataset
│   ├── interpolate_scaled_profile()  # Profile processing
│   ├── compute_enhanced_features()   # Feature engineering
│   ├── build_dataset()      # Complete dataset construction
│   ├── train_final_model()  # Model training & validation
│   ├── predict()           # Single shot prediction
│   └── plot_feature_importance()    # Visualization
└── main()                  # Complete pipeline demo
```

## Physics Background

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

## Future Work

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

## References

1. **ITER Physics Basis** - Progress in the ITER Physics Basis, Nuclear Fusion (2007)
2. **H-mode Discovery** - Wagner et al., Physical Review Letters (1982)
3. **DIII-D Thomson Scattering** - Eldon et al., Review of Scientific Instruments (2018)
4. **Machine Learning in Fusion** - Kates-Harbeck et al., Nature (2019)

## Contributing

We welcome contributions! Areas of interest:
- Feature engineering improvements
- Alternative ML architectures  
- Cross-machine validation
- Real-time implementation
- Physics-informed constraints

## License

MIT License

## Acknowledgments

- **DIII-D National Fusion Facility**: Dr Vacslav Glukhov and Dr Randall Clark for their incredible patience and guidance, as well as for providing the labelled experimental data
- **University College London**: Dr Dmitry Adamskiy for supporting this research and providing guidance and oversight throughout

---

**Citation**: If you use this code in your research, please cite:
```
@misc{hmode_classifier_2024,
  title={Enhanced Machine Learning Classification of H-mode Confinement in Tokamak Plasmas},
  author={MSc Plasma Physics Project},
  year={2024},
  url={https://github.com/despinademetriadou/hmode-classifier}
}
```
