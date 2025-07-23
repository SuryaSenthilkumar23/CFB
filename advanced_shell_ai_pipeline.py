#!/usr/bin/env python3
"""
Shell.ai Hackathon 2025 - ADVANCED PIPELINE
===========================================

Complete advanced solution with:
- CatBoost, LightGBM, XGBoost models optimized for MAE/MAPE
- Property-wise modeling (separate models per target)
- Advanced feature engineering with interactions
- Stacking/blending with meta-regressor
- Robust cross-validation with logging
- Hyperparameter optimization
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Advanced models
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️ LightGBM not available, using alternatives")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, using alternatives")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using alternatives")

import json
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for fuel blend properties"""
    
    def __init__(self):
        self.feature_names = []
        self.component_cols = []
        self.property_cols = []
        
    def fit(self, X):
        """Fit the feature engineer on training data"""
        self.component_cols = [col for col in X.columns if 'fraction' in col.lower()]
        self.property_cols = [col for col in X.columns if 'Property' in col]
        
        logger.info(f"Found {len(self.component_cols)} component columns")
        logger.info(f"Found {len(self.property_cols)} property columns")
        
        return self
    
    def transform(self, X):
        """Transform data with advanced feature engineering"""
        X_new = X.copy()
        
        # 1. Basic statistics for components and properties
        if self.component_cols:
            comp_data = X[self.component_cols]
            X_new['comp_mean'] = comp_data.mean(axis=1)
            X_new['comp_std'] = comp_data.std(axis=1)
            X_new['comp_min'] = comp_data.min(axis=1)
            X_new['comp_max'] = comp_data.max(axis=1)
            X_new['comp_median'] = comp_data.median(axis=1)
            X_new['comp_sum'] = comp_data.sum(axis=1)
            X_new['comp_range'] = X_new['comp_max'] - X_new['comp_min']
            X_new['comp_cv'] = X_new['comp_std'] / (X_new['comp_mean'] + 1e-8)
        
        if self.property_cols:
            prop_data = X[self.property_cols]
            X_new['prop_mean'] = prop_data.mean(axis=1)
            X_new['prop_std'] = prop_data.std(axis=1)
            X_new['prop_min'] = prop_data.min(axis=1)
            X_new['prop_max'] = prop_data.max(axis=1)
            X_new['prop_median'] = prop_data.median(axis=1)
            X_new['prop_range'] = X_new['prop_max'] - X_new['prop_min']
            X_new['prop_cv'] = X_new['prop_std'] / (X_new['prop_mean'] + 1e-8)
            X_new['prop_skew'] = prop_data.skew(axis=1)
            X_new['prop_kurt'] = prop_data.kurtosis(axis=1)
        
        # 2. Weighted averages of blend components with their properties
        self._add_weighted_features(X, X_new)
        
        # 3. Component-property interactions
        self._add_interaction_features(X, X_new)
        
        # 4. Polynomial features for top components
        self._add_polynomial_features(X, X_new)
        
        # 5. Transformed features
        self._add_transformed_features(X, X_new)
        
        # 6. Cross-component ratios and products
        self._add_cross_component_features(X, X_new)
        
        return X_new
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def _add_weighted_features(self, X, X_new):
        """Add weighted average features"""
        # Map components to their properties
        component_map = {}
        for i in range(1, 6):  # Components 1-5
            frac_col = None
            prop_cols = []
            
            for col in X.columns:
                if f'Component{i}_fraction' in col:
                    frac_col = col
                elif f'Component{i}' in col and 'Property' in col:
                    prop_cols.append(col)
            
            if frac_col and prop_cols:
                component_map[i] = {'fraction': frac_col, 'properties': prop_cols}
        
        # Create weighted features
        for comp_id, comp_data in component_map.items():
            frac_col = comp_data['fraction']
            prop_cols = comp_data['properties']
            
            if frac_col in X.columns and len(prop_cols) > 0:
                fraction = X[frac_col]
                
                # Weighted average of properties for this component
                for prop_col in prop_cols[:10]:  # Limit to avoid too many features
                    if prop_col in X.columns:
                        X_new[f'weighted_{comp_id}_{prop_col.split("_")[-1]}'] = fraction * X[prop_col]
                
                # Component-wise aggregates
                comp_props = X[prop_cols]
                X_new[f'comp{comp_id}_prop_mean'] = comp_props.mean(axis=1)
                X_new[f'comp{comp_id}_prop_std'] = comp_props.std(axis=1)
                X_new[f'comp{comp_id}_prop_max'] = comp_props.max(axis=1)
                X_new[f'comp{comp_id}_prop_min'] = comp_props.min(axis=1)
                
                # Weighted component statistics
                X_new[f'comp{comp_id}_weighted_mean'] = fraction * comp_props.mean(axis=1)
                X_new[f'comp{comp_id}_intensity'] = comp_props.mean(axis=1) / (fraction + 1e-8)
    
    def _add_interaction_features(self, X, X_new):
        """Add interaction features between components and properties"""
        # Component × Component interactions
        comp_fractions = [col for col in self.component_cols if col in X.columns]
        
        for i, comp1 in enumerate(comp_fractions):
            for comp2 in comp_fractions[i+1:]:
                X_new[f'{comp1}_x_{comp2}'] = X[comp1] * X[comp2]
                X_new[f'{comp1}_ratio_{comp2}'] = X[comp1] / (X[comp2] + 1e-8)
        
        # Component × Property interactions (top properties only)
        top_properties = self.property_cols[:20] if len(self.property_cols) > 20 else self.property_cols
        
        for comp in comp_fractions[:3]:  # Top 3 components
            for prop in top_properties[:5]:  # Top 5 properties
                if comp in X.columns and prop in X.columns:
                    X_new[f'{comp}_x_{prop}'] = X[comp] * X[prop]
    
    def _add_polynomial_features(self, X, X_new):
        """Add polynomial features for key components"""
        key_components = self.component_cols[:5] if len(self.component_cols) > 5 else self.component_cols
        
        for comp in key_components:
            if comp in X.columns:
                X_new[f'{comp}_squared'] = X[comp] ** 2
                X_new[f'{comp}_cubed'] = X[comp] ** 3
                X_new[f'{comp}_sqrt'] = np.sqrt(np.abs(X[comp]))
    
    def _add_transformed_features(self, X, X_new):
        """Add transformed features"""
        # Log transforms for positive features
        for col in self.component_cols + self.property_cols[:10]:
            if col in X.columns:
                X_new[f'{col}_log1p'] = np.log1p(np.abs(X[col]))
                X_new[f'{col}_reciprocal'] = 1 / (np.abs(X[col]) + 1e-8)
    
    def _add_cross_component_features(self, X, X_new):
        """Add cross-component features"""
        if len(self.component_cols) >= 2:
            comp_data = X[self.component_cols]
            
            # Diversity measures
            X_new['comp_entropy'] = -np.sum(comp_data * np.log(comp_data + 1e-8), axis=1)
            X_new['comp_gini'] = 1 - np.sum(comp_data ** 2, axis=1)
            
            # Dominant component features
            X_new['dominant_comp'] = comp_data.idxmax(axis=1).astype('category').cat.codes
            X_new['dominant_comp_value'] = comp_data.max(axis=1)
            X_new['second_comp_value'] = comp_data.apply(lambda x: x.nlargest(2).iloc[1], axis=1)


class PropertyWiseModelManager:
    """Manages separate models for each target property"""
    
    def __init__(self, target_names):
        self.target_names = target_names
        self.models = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    def get_base_models(self):
        """Get base models for stacking"""
        models = []
        
        # Ridge Regression
        models.append(('ridge', Ridge(alpha=1.0, random_state=42)))
        
        # SVR
        models.append(('svr', SVR(kernel='rbf', C=1.0, gamma='scale')))
        
        # Advanced models if available
        if LGBM_AVAILABLE:
            models.append(('lgbm', lgb.LGBMRegressor(
                objective='mae',
                metric='mae',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42,
                n_estimators=500
            )))
        
        if CATBOOST_AVAILABLE:
            models.append(('catboost', cb.CatBoostRegressor(
                loss_function='MAE',
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )))
        
        if XGBOOST_AVAILABLE:
            models.append(('xgboost', xgb.XGBRegressor(
                objective='reg:absoluteerror',
                eval_metric='mae',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )))
        
        return models
    
    def train_property_model(self, X_train, y_train, property_name):
        """Train stacked model for a specific property"""
        logger.info(f"Training model for {property_name}")
        
        # Get base models
        base_models = self.get_base_models()
        
        # Meta-regressor
        meta_regressor = HuberRegressor(epsilon=1.35, alpha=0.01)
        
        # Create stacking regressor
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_regressor,
            cv=5,
            n_jobs=-1,
            passthrough=False
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            stacking_model, X_train, y_train,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        # Train final model
        stacking_model.fit(X_train, y_train)
        
        # Store results
        self.models[property_name] = stacking_model
        self.cv_scores[property_name] = {
            'mae_mean': -cv_scores.mean(),
            'mae_std': cv_scores.std(),
            'scores': -cv_scores
        }
        
        logger.info(f"{property_name} - CV MAE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return stacking_model
    
    def predict_property(self, X_test, property_name):
        """Predict for a specific property"""
        if property_name not in self.models:
            raise ValueError(f"Model for {property_name} not trained")
        
        return self.models[property_name].predict(X_test)
    
    def predict_all(self, X_test):
        """Predict all properties"""
        predictions = pd.DataFrame()
        
        for property_name in self.target_names:
            predictions[property_name] = self.predict_property(X_test, property_name)
        
        return predictions


class ModelLogger:
    """Logs model performance and settings"""
    
    def __init__(self, log_dir='model_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def log_experiment(self, cv_scores, feature_count, model_config, notes=""):
        """Log experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        experiment_data = {
            'timestamp': timestamp,
            'cv_scores': cv_scores,
            'feature_count': feature_count,
            'model_config': model_config,
            'notes': notes,
            'avg_mae': np.mean([scores['mae_mean'] for scores in cv_scores.values()]),
            'total_mae_std': np.mean([scores['mae_std'] for scores in cv_scores.values()])
        }
        
        # Save to JSON
        log_file = os.path.join(self.log_dir, f'experiment_{timestamp}.json')
        with open(log_file, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        logger.info(f"Experiment logged to {log_file}")
        logger.info(f"Average MAE across properties: {experiment_data['avg_mae']:.4f}")
        
        return experiment_data


def load_data():
    """Load and validate data"""
    logger.info("Loading data...")
    
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sample_df = pd.read_csv('sample_solution.csv')
        
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")
        logger.info(f"Sample shape: {sample_df.shape}")
        
        return train_df, test_df, sample_df
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise


def prepare_features_targets(train_df):
    """Prepare features and targets"""
    logger.info("Preparing features and targets...")
    
    # Identify target columns
    target_cols = [col for col in train_df.columns if 'BlendProperty' in col]
    feature_cols = [col for col in train_df.columns if col not in target_cols + ['ID']]
    
    logger.info(f"Found {len(target_cols)} target columns")
    logger.info(f"Found {len(feature_cols)} feature columns")
    
    X = train_df[feature_cols]
    y = train_df[target_cols]
    
    return X, y, target_cols, feature_cols


def create_submission(predictions, test_df, target_cols, filename='submission.csv'):
    """Create submission file"""
    logger.info("Creating submission file...")
    
    submission = pd.DataFrame()
    submission['ID'] = test_df['ID']
    
    for col in target_cols:
        submission[col] = predictions[col]
    
    submission.to_csv(filename, index=False)
    logger.info(f"Submission saved to {filename}")
    
    # Log prediction statistics
    for col in target_cols:
        mean_pred = predictions[col].mean()
        std_pred = predictions[col].std()
        min_pred = predictions[col].min()
        max_pred = predictions[col].max()
        
        logger.info(f"{col}: mean={mean_pred:.3f}, std={std_pred:.3f}, range=[{min_pred:.3f}, {max_pred:.3f}]")


def main():
    """Main pipeline execution"""
    logger.info("🚀 Starting Advanced Shell.ai Pipeline")
    logger.info("=" * 60)
    
    try:
        # Load data
        train_df, test_df, sample_df = load_data()
        
        # Prepare features and targets
        X_train, y_train, target_cols, feature_cols = prepare_features_targets(train_df)
        X_test = test_df[feature_cols]
        
        # Advanced feature engineering
        logger.info("🔧 Advanced Feature Engineering")
        feature_engineer = AdvancedFeatureEngineer()
        X_train_engineered = feature_engineer.fit_transform(X_train)
        X_test_engineered = feature_engineer.transform(X_test)
        
        logger.info(f"Features: {X_train.shape[1]} → {X_train_engineered.shape[1]} (+{X_train_engineered.shape[1] - X_train.shape[1]})")
        
        # Handle missing values and infinite values
        X_train_engineered = X_train_engineered.replace([np.inf, -np.inf], np.nan)
        X_test_engineered = X_test_engineered.replace([np.inf, -np.inf], np.nan)
        
        X_train_engineered = X_train_engineered.fillna(X_train_engineered.median())
        X_test_engineered = X_test_engineered.fillna(X_train_engineered.median())
        
        # Feature scaling
        logger.info("📊 Feature Scaling")
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_engineered),
            columns=X_train_engineered.columns,
            index=X_train_engineered.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_engineered),
            columns=X_test_engineered.columns,
            index=X_test_engineered.index
        )
        
        # Property-wise modeling
        logger.info("🎯 Property-wise Modeling")
        model_manager = PropertyWiseModelManager(target_cols)
        
        # Train models for each property
        for target_col in target_cols:
            model_manager.train_property_model(
                X_train_scaled, 
                y_train[target_col], 
                target_col
            )
        
        # Make predictions
        logger.info("🔮 Making Predictions")
        predictions = model_manager.predict_all(X_test_scaled)
        
        # Log experiment
        logger.info("📝 Logging Experiment")
        model_logger = ModelLogger()
        
        model_config = {
            'base_models': ['ridge', 'svr'] + 
                          (['lgbm'] if LGBM_AVAILABLE else []) +
                          (['catboost'] if CATBOOST_AVAILABLE else []) +
                          (['xgboost'] if XGBOOST_AVAILABLE else []),
            'meta_regressor': 'HuberRegressor',
            'feature_engineering': 'Advanced with interactions',
            'scaling': 'RobustScaler',
            'cv_folds': 5
        }
        
        experiment_data = model_logger.log_experiment(
            model_manager.cv_scores,
            X_train_scaled.shape[1],
            model_config,
            "Advanced pipeline with property-wise modeling"
        )
        
        # Create submission
        create_submission(predictions, test_df, target_cols)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("🎉 ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Features engineered: {X_train_scaled.shape[1]}")
        logger.info(f"🎯 Properties modeled: {len(target_cols)}")
        logger.info(f"📈 Average CV MAE: {experiment_data['avg_mae']:.4f}")
        logger.info(f"📁 Submission: submission.csv")
        logger.info("=" * 60)
        
        return experiment_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Install required packages if needed
    import subprocess
    import sys
    
    def install_package(package):
        try:
            __import__(package)
        except ImportError:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Try to install missing packages
    packages_to_try = ['lightgbm', 'catboost', 'xgboost']
    for package in packages_to_try:
        try:
            install_package(package)
        except:
            logger.warning(f"Could not install {package}, will use alternatives")
    
    # Run main pipeline
    main()