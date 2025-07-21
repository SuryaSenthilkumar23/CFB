# 🏆 Shell.ai Hackathon Level 1: Fuel Blend Properties Prediction

**Elite ML Pipeline for Winning the Shell.ai Hackathon**

## 🎯 Challenge Overview

**Goal**: Predict 10 blend properties based on:
- 5 component percentages  
- 50 component-specific properties (10 per component)

**Metric**: MAPE (Mean Absolute Percentage Error)  
**Strategy**: Minimize average MAPE across all 10 target properties

## 📊 Data Structure

- **train.csv**: 55 input features + 10 targets (BlendProperty1 to BlendProperty10)
- **test.csv**: 55 input features (same structure as train)  
- **sample_solution.csv**: Submission format example

## 🚀 Solution Architecture

### 1. Feature Engineering Pipeline
- **Component Normalization**: Ensure percentages sum to 100%
- **Property Aggregations**: Mean, std, min, max per component
- **Component Interactions**: Multiplication and division ratios
- **Weighted Features**: Component % × Property values
- **PCA Features**: Dimensionality reduction for properties
- **Zero Variance Removal**: Clean feature space

### 2. Model Ensemble Strategy
- **XGBoost**: Gradient boosting with tree-based learning
- **LightGBM**: Fast gradient boosting with leaf-wise growth
- **CatBoost**: Gradient boosting with categorical feature handling
- **Voting Ensemble**: Equal-weight combination of all models

### 3. Hyperparameter Optimization
- **Optuna**: Bayesian optimization for each model
- **Cross-Validation**: 5-fold CV with MAPE scoring
- **Target-Specific**: Separate optimization per blend property

### 4. Training Pipeline
- **Robust Scaling**: Handle outliers in feature scaling
- **Multi-Target**: Train separate ensemble per target property
- **Feature Importance**: Track important features per model
- **Performance Monitoring**: CV and training MAPE tracking

## 📁 Project Structure

```
├── shell_ai_fuel_blend_predictor.py    # Main pipeline script
├── Shell_AI_Hackathon_Level1.ipynb     # Interactive notebook
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── train.csv                          # Training data (place here)
├── test.csv                           # Test data (place here)
├── sample_solution.csv                # Submission format (place here)
└── outputs/
    ├── submission.csv                 # Final predictions
    ├── feature_importance.png         # Feature analysis plots
    ├── model_performance.png          # Performance visualizations
    └── model_insights.json           # Detailed metrics
```

## 🔧 Installation & Setup

### Option 1: Using pip (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
conda install pandas numpy scikit-learn xgboost lightgbm catboost optuna matplotlib seaborn jupyter
```

### Option 3: System packages (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3-pandas python3-numpy python3-sklearn python3-matplotlib python3-seaborn
pip install --user xgboost lightgbm catboost optuna
```

## 🚀 Quick Start

### 1. Run Complete Pipeline
```bash
python shell_ai_fuel_blend_predictor.py
```

### 2. Interactive Development
```bash
jupyter notebook Shell_AI_Hackathon_Level1.ipynb
```

### 3. Quick Demo (with sample data)
```python
from shell_ai_fuel_blend_predictor import FuelBlendPredictor, create_sample_data

# Create sample data if real data not available
create_sample_data()

# Initialize and run pipeline
predictor = FuelBlendPredictor(random_state=42)
# ... (see notebook for full pipeline)
```

## ⚙️ Configuration Options

### Hyperparameter Optimization
```python
# Fast baseline (demo mode)
predictor.train_ensemble_models(X_train, y_train, optimize_params=False)

# Full optimization (production mode)
predictor.train_ensemble_models(X_train, y_train, optimize_params=True)
```

### Model Selection
```python
# Custom model selection
models = {
    'xgboost': xgb.XGBRegressor(**custom_params),
    'lightgbm': lgb.LGBMRegressor(**custom_params),
    'catboost': cb.CatBoostRegressor(**custom_params)
}
```

## 📈 Expected Performance

### Baseline Performance (without optimization)
- **Average MAPE**: ~0.05-0.15 (depends on data complexity)
- **Training Time**: ~5-10 minutes
- **Memory Usage**: ~1-2 GB

### Optimized Performance (with hyperparameter tuning)
- **Average MAPE**: ~0.03-0.10 (typically 20-30% improvement)
- **Training Time**: ~30-60 minutes
- **Memory Usage**: ~2-4 GB

## 🎯 Key Features

### ✅ Production-Ready
- Modular, reusable code architecture
- Comprehensive error handling
- Automated feature engineering
- Cross-validation with proper metrics

### ✅ Hackathon-Optimized
- Fast baseline for quick iterations
- Hyperparameter optimization for final submission
- Feature importance analysis
- Performance visualization

### ✅ Extensible Design
- Easy to add new models
- Configurable feature engineering
- Pluggable optimization strategies
- Multi-target learning support

## 🔍 Feature Engineering Details

### Component Features
```python
# Normalized percentages
'%Component1_normalized', '%Component2_normalized', ...

# Component interactions  
'%Component1_x_%Component2', '%Component1_div_%Component2', ...

# Weighted properties
'%Component1_weighted_Component1_Property1', ...
```

### Property Features
```python
# Aggregated statistics per component
'Component1_mean', 'Component1_std', 'Component1_min', 'Component1_max'

# PCA components
'PCA_Property_1', 'PCA_Property_2', ...
```

## 📊 Model Performance Analysis

### Cross-Validation Strategy
- **5-Fold CV**: Robust performance estimation
- **MAPE Scoring**: Directly optimizes competition metric
- **Target-Specific**: Separate CV per blend property

### Ensemble Strategy
- **Voting Regressor**: Equal weight combination
- **Model Diversity**: Different algorithms for robust predictions
- **Feature Importance**: Track influential features

## 💡 Advanced Optimization Ideas

### 1. Hyperparameter Tuning
```python
# Increase optimization trials
predictor.optimize_hyperparameters(X, y, 'xgboost', n_trials=200)
```

### 2. Advanced Ensembling
```python
# Stacking ensemble
from sklearn.ensemble import StackingRegressor
stacking_ensemble = StackingRegressor(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
    final_estimator=LinearRegression()
)
```

### 3. Feature Selection
```python
# Recursive feature elimination
from sklearn.feature_selection import RFE
selector = RFE(estimator=xgb_model, n_features_to_select=50)
```

### 4. Multi-Target Learning
```python
# Multi-output regressor
from sklearn.multioutput import MultiOutputRegressor
multi_target_model = MultiOutputRegressor(xgb.XGBRegressor())
```

## 🐛 Troubleshooting

### Common Issues

**1. Memory Errors**
```python
# Reduce feature engineering complexity
# Use smaller n_estimators
# Process targets sequentially
```

**2. Slow Training**
```python
# Set optimize_params=False for quick baseline
# Reduce n_trials in optimization
# Use early stopping
```

**3. Poor Performance**
```python
# Check data quality (nulls, outliers)
# Verify feature engineering logic
# Increase model complexity
# Add more feature interactions
```

## 📝 Submission Checklist

- [ ] Data files (train.csv, test.csv) in project directory
- [ ] Run complete pipeline: `python shell_ai_fuel_blend_predictor.py`
- [ ] Verify submission.csv format matches sample_solution.csv
- [ ] Check average MAPE < 0.15 (reasonable baseline)
- [ ] Review feature importance plots for insights
- [ ] Test hyperparameter optimization if time permits

## 🏆 Competition Strategy

### Phase 1: Quick Baseline (30 minutes)
1. Run pipeline with `optimize_params=False`
2. Submit initial predictions
3. Analyze feature importance

### Phase 2: Feature Engineering (60 minutes)
1. Add domain-specific features
2. Feature selection experiments
3. Cross-validation analysis

### Phase 3: Model Optimization (90 minutes)
1. Hyperparameter tuning with Optuna
2. Advanced ensembling techniques
3. Final model selection

### Phase 4: Final Submission (30 minutes)
1. Generate final predictions
2. Verify submission format
3. Submit with confidence!

## 🤝 Contributing

Feel free to enhance this solution with:
- Advanced feature engineering techniques
- New model architectures (Neural Networks, etc.)
- Improved ensemble strategies
- Better hyperparameter optimization

## 📄 License

This project is designed for the Shell.ai Hackathon. Use responsibly and good luck! 🚀

---

**Built with ❤️ for Shell.ai Hackathon Level 1**

*Ready to win? Let's fuel the future! ⛽🏆*