#!/usr/bin/env python3
"""
Shell.ai Hackathon 2025 - Advanced Stacked Ensemble Solution
============================================================

High-performance stacked ensemble for fuel blend property prediction.
Target: Maximize private leaderboard score (goal: >97)

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any
import joblib
from pathlib import Path

# Core ML libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import (
    StackingRegressor, RandomForestRegressor, 
    GradientBoostingRegressor
)
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression

# Gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Visualization and utilities
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)

class ShellAIEnsemble:
    """
    Advanced Stacked Ensemble for Shell.ai Hackathon 2025
    
    Features:
    - 4 Base models: XGBoost, LightGBM, CatBoost, RandomForest
    - Meta-learner: RidgeCV with cross-validation
    - 5-Fold cross-validation for robust training
    - Feature engineering and selection
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, random_state: int = 42, n_folds: int = 5):
        self.random_state = random_state
        self.n_folds = n_folds
        self.scaler = None
        self.feature_selector = None
        self.stacking_regressor = None
        self.feature_names = None
        self.target_name = None
        self.cv_scores = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self, train_path: str, test_path: str, sample_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the dataset"""
        print("🔄 Loading datasets...")
        
        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        sample_df = pd.read_csv(sample_path)
        
        print(f"📊 Dataset shapes:")
        print(f"   Train: {train_df.shape}")
        print(f"   Test: {test_df.shape}")
        print(f"   Sample: {sample_df.shape}")
        
        # Basic data quality checks
        print(f"\n🔍 Data Quality Check:")
        print(f"   Train missing values: {train_df.isnull().sum().sum()}")
        print(f"   Test missing values: {test_df.isnull().sum().sum()}")
        
        # Handle missing values if any
        if train_df.isnull().sum().sum() > 0:
            print("⚠️  Handling missing values in training data...")
            train_df = train_df.fillna(train_df.median(numeric_only=True))
            
        if test_df.isnull().sum().sum() > 0:
            print("⚠️  Handling missing values in test data...")
            test_df = test_df.fillna(test_df.median(numeric_only=True))
        
        return train_df, test_df, sample_df
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Advanced feature engineering for fuel blend prediction"""
        print("🔧 Engineering features...")
        
        def add_features(df):
            df_new = df.copy()
            
            # Identify feature columns (exclude ID and target if present)
            feature_cols = [col for col in df.columns if col not in ['ID', 'id'] and not col.startswith('target')]
            if len(feature_cols) == 0:
                feature_cols = [col for col in df.columns if col not in ['ID', 'id']]
            
            # Component percentage features (assuming they exist)
            component_cols = [col for col in feature_cols if 'component' in col.lower() or '%' in col.lower()]
            property_cols = [col for col in feature_cols if col not in component_cols]
            
            if len(component_cols) > 0:
                # Component-based features
                df_new['total_components'] = df_new[component_cols].sum(axis=1)
                df_new['max_component'] = df_new[component_cols].max(axis=1)
                df_new['min_component'] = df_new[component_cols].min(axis=1)
                df_new['component_range'] = df_new['max_component'] - df_new['min_component']
                df_new['component_std'] = df_new[component_cols].std(axis=1)
                df_new['component_cv'] = df_new['component_std'] / (df_new[component_cols].mean(axis=1) + 1e-8)
                
                # Dominant component features
                df_new['dominant_component_idx'] = df_new[component_cols].idxmax(axis=1)
                df_new['dominant_component_value'] = df_new[component_cols].max(axis=1)
                
                # Component ratios (top 3 components)
                if len(component_cols) >= 2:
                    sorted_components = df_new[component_cols].apply(lambda x: x.sort_values(ascending=False), axis=1)
                    df_new['comp_ratio_1_2'] = sorted_components.iloc[:, 0] / (sorted_components.iloc[:, 1] + 1e-8)
                    if len(component_cols) >= 3:
                        df_new['comp_ratio_1_3'] = sorted_components.iloc[:, 0] / (sorted_components.iloc[:, 2] + 1e-8)
                        df_new['comp_ratio_2_3'] = sorted_components.iloc[:, 1] / (sorted_components.iloc[:, 2] + 1e-8)
            
            if len(property_cols) > 0:
                # Property-based features
                df_new['total_properties'] = df_new[property_cols].sum(axis=1)
                df_new['mean_properties'] = df_new[property_cols].mean(axis=1)
                df_new['max_property'] = df_new[property_cols].max(axis=1)
                df_new['min_property'] = df_new[property_cols].min(axis=1)
                df_new['property_range'] = df_new['max_property'] - df_new['min_property']
                df_new['property_std'] = df_new[property_cols].std(axis=1)
                df_new['property_cv'] = df_new['property_std'] / (df_new['mean_properties'] + 1e-8)
                df_new['property_skew'] = df_new[property_cols].skew(axis=1)
                df_new['property_kurtosis'] = df_new[property_cols].kurtosis(axis=1)
            
            # Interaction features between components and properties
            if len(component_cols) > 0 and len(property_cols) > 0:
                for i, comp_col in enumerate(component_cols[:3]):  # Top 3 components
                    for j, prop_col in enumerate(property_cols[:5]):  # Top 5 properties
                        df_new[f'{comp_col}_x_{prop_col}'] = df_new[comp_col] * df_new[prop_col]
                        if i == 0 and j < 3:  # More interactions for dominant component
                            df_new[f'{comp_col}_div_{prop_col}'] = df_new[comp_col] / (df_new[prop_col] + 1e-8)
            
            # Polynomial features for key variables
            numeric_cols = df_new.select_dtypes(include=[np.number]).columns
            key_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                       for keyword in ['component', 'property', 'blend'])][:5]
            
            for col in key_cols:
                if df_new[col].std() > 0:  # Only if there's variance
                    df_new[f'{col}_squared'] = df_new[col] ** 2
                    df_new[f'{col}_sqrt'] = np.sqrt(np.abs(df_new[col]))
                    df_new[f'{col}_log'] = np.log1p(np.abs(df_new[col]))
            
            return df_new
        
        # Apply feature engineering
        train_enhanced = add_features(train_df)
        test_enhanced = add_features(test_df)
        
        print(f"✅ Feature engineering complete:")
        print(f"   Original features: {train_df.shape[1]}")
        print(f"   Enhanced features: {train_enhanced.shape[1]}")
        
        return train_enhanced, test_enhanced
    
    def prepare_features_and_target(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and target for modeling"""
        print("📋 Preparing features and target...")
        
        # Identify target column (assuming it's the last column or contains 'target')
        target_candidates = [col for col in train_df.columns if 'target' in col.lower() or 'blend' in col.lower()]
        if target_candidates:
            self.target_name = target_candidates[0]
        else:
            # Assume last column is target
            self.target_name = train_df.columns[-1]
        
        print(f"🎯 Target column: {self.target_name}")
        
        # Prepare features
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'id', self.target_name]]
        self.feature_names = feature_cols
        
        X_train = train_df[feature_cols].values
        y_train = train_df[self.target_name].values
        X_test = test_df[feature_cols].values
        
        print(f"📊 Final dataset shapes:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   X_test: {X_test.shape}")
        
        # Handle infinite values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return X_train, y_train, X_test
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using RobustScaler"""
        print("⚖️  Scaling features...")
        
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k_best: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Feature selection using SelectKBest"""
        if k_best is None:
            k_best = min(50, X_train.shape[1])  # Select top 50 or all features
        
        print(f"🎯 Selecting top {k_best} features...")
        
        self.feature_selector = SelectKBest(score_func=f_regression, k=k_best)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        print(f"✅ Selected features: {len(selected_features)}")
        
        return X_train_selected, X_test_selected
    
    def create_base_models(self) -> Dict[str, Any]:
        """Create optimized base models"""
        print("🏗️  Creating base models...")
        
        base_models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                early_stopping_rounds=50,
                eval_metric='rmse'
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                force_col_wise=True,
                verbose=-1
            ),
            
            'catboost': cb.CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.05,
                subsample=0.8,
                reg_lambda=1.0,
                random_state=self.random_state,
                thread_count=-1,
                verbose=False,
                early_stopping_rounds=50
            ),
            
            'randomforest': RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'gradientboosting': GradientBoostingRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            )
        }
        
        print(f"✅ Created {len(base_models)} base models")
        return base_models
    
    def create_meta_model(self) -> Any:
        """Create meta-learner model"""
        print("🧠 Creating meta-learner...")
        
        # Try both Ridge and Lasso, use the better one
        ridge_model = RidgeCV(
            alphas=np.logspace(-3, 3, 50),
            cv=self.n_folds,
            scoring='neg_mean_squared_error'
        )
        
        return ridge_model
    
    def evaluate_base_models(self, base_models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Evaluate base models using cross-validation"""
        print("📊 Evaluating base models with cross-validation...")
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in tqdm(base_models.items(), desc="Evaluating models"):
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=kfold, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                rmse_scores = np.sqrt(-cv_scores)
                self.cv_scores[name] = {
                    'rmse_mean': rmse_scores.mean(),
                    'rmse_std': rmse_scores.std(),
                    'scores': rmse_scores
                }
                
                print(f"   {name:15} - RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {name}: {str(e)}")
                self.cv_scores[name] = {'rmse_mean': float('inf'), 'rmse_std': 0, 'scores': []}
    
    def train_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the stacking ensemble"""
        print("🏗️  Training stacking ensemble...")
        
        # Create base models
        base_models = self.create_base_models()
        
        # Evaluate base models first
        self.evaluate_base_models(base_models, X_train, y_train)
        
        # Select best performing base models (top 4)
        sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['rmse_mean'])
        best_model_names = [name for name, _ in sorted_models[:4]]
        selected_base_models = [(name, base_models[name]) for name in best_model_names]
        
        print(f"🎯 Selected base models: {best_model_names}")
        
        # Create meta-model
        meta_model = self.create_meta_model()
        
        # Create stacking regressor
        self.stacking_regressor = StackingRegressor(
            estimators=selected_base_models,
            final_estimator=meta_model,
            cv=self.n_folds,
            n_jobs=-1,
            verbose=1
        )
        
        # Train the ensemble
        print("🚀 Training stacking ensemble...")
        self.stacking_regressor.fit(X_train, y_train)
        
        # Get feature importance from base models
        self.extract_feature_importance(selected_base_models, X_train, y_train)
        
        print("✅ Stacking ensemble training complete!")
    
    def extract_feature_importance(self, base_models: List[Tuple[str, Any]], X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Extract feature importance from base models"""
        print("📊 Extracting feature importance...")
        
        for name, model in base_models:
            try:
                # Fit model to get feature importance
                model.fit(X_train, y_train)
                
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = np.abs(model.coef_)
                    
            except Exception as e:
                print(f"   ⚠️  Could not extract importance for {name}: {str(e)}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using the trained ensemble"""
        print("🔮 Making predictions...")
        
        if self.stacking_regressor is None:
            raise ValueError("Model not trained yet. Call train_stacking_ensemble first.")
        
        predictions = self.stacking_regressor.predict(X_test)
        return predictions
    
    def evaluate_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Evaluate the ensemble on training data"""
        print("📊 Evaluating ensemble performance...")
        
        # Cross-validation evaluation
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            self.stacking_regressor, X_train, y_train,
            cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        cv_rmse = np.sqrt(-cv_scores)
        
        # Training predictions for additional metrics
        train_pred = self.stacking_regressor.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        try:
            train_mape = mean_absolute_percentage_error(y_train, train_pred)
        except:
            train_mape = np.mean(np.abs((y_train - train_pred) / (y_train + 1e-8))) * 100
        
        metrics = {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'train_rmse': train_rmse,
            'train_mape': train_mape
        }
        
        print(f"📊 Ensemble Performance:")
        print(f"   CV RMSE: {metrics['cv_rmse_mean']:.4f} (±{metrics['cv_rmse_std']:.4f})")
        print(f"   Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"   Train MAPE: {metrics['train_mape']:.2f}%")
        
        return metrics
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance from base models"""
        if not self.feature_importance:
            print("⚠️  No feature importance data available")
            return
        
        print(f"📊 Plotting top {top_n} feature importance...")
        
        # Get selected feature names
        if self.feature_selector:
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_feature_names = [self.feature_names[i] for i in selected_indices]
        else:
            selected_feature_names = self.feature_names
        
        # Create subplots
        n_models = len(self.feature_importance)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (model_name, importance) in enumerate(self.feature_importance.items()):
            if idx >= 4:  # Only plot first 4 models
                break
                
            # Get top features
            top_indices = np.argsort(importance)[-top_n:]
            top_importance = importance[top_indices]
            top_features = [selected_feature_names[i] for i in top_indices]
            
            # Plot
            axes[idx].barh(range(len(top_features)), top_importance)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features, fontsize=8)
            axes[idx].set_title(f'{model_name.title()} - Feature Importance')
            axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature importance plot saved as 'feature_importance.png'")
    
    def save_model(self, filepath: str = 'shell_ai_ensemble_model.pkl') -> None:
        """Save the trained model"""
        print(f"💾 Saving model to {filepath}...")
        
        model_data = {
            'stacking_regressor': self.stacking_regressor,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print("✅ Model saved successfully!")
    
    def create_submission(self, predictions: np.ndarray, test_df: pd.DataFrame, sample_df: pd.DataFrame, output_path: str = 'submission.csv') -> None:
        """Create submission file matching the sample format"""
        print("📝 Creating submission file...")
        
        # Create submission dataframe
        submission = sample_df.copy()
        
        # Identify ID column
        id_col = 'ID' if 'ID' in test_df.columns else 'id'
        target_col = [col for col in sample_df.columns if col != id_col][0]
        
        # Fill predictions
        submission[id_col] = test_df[id_col]
        submission[target_col] = predictions
        
        # Save submission
        submission.to_csv(output_path, index=False)
        
        print(f"✅ Submission saved to {output_path}")
        print(f"📊 Submission statistics:")
        print(f"   Shape: {submission.shape}")
        print(f"   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"   Predictions mean: {predictions.mean():.4f}")
        print(f"   Predictions std: {predictions.std():.4f}")


def main():
    """Main execution function"""
    print("🚀 Shell.ai Hackathon 2025 - Advanced Stacked Ensemble")
    print("=" * 60)
    
    # Initialize ensemble
    ensemble = ShellAIEnsemble(random_state=42, n_folds=5)
    
    try:
        # Load and preprocess data
        train_df, test_df, sample_df = ensemble.load_and_preprocess_data(
            'train.csv', 'test.csv', 'sample_solution.csv'
        )
        
        # Feature engineering
        train_enhanced, test_enhanced = ensemble.engineer_features(train_df, test_df)
        
        # Prepare features and target
        X_train, y_train, X_test = ensemble.prepare_features_and_target(train_enhanced, test_enhanced)
        
        # Scale features
        X_train_scaled, X_test_scaled = ensemble.scale_features(X_train, X_test)
        
        # Feature selection
        X_train_final, X_test_final = ensemble.select_features(X_train_scaled, y_train, X_test_scaled)
        
        # Train stacking ensemble
        ensemble.train_stacking_ensemble(X_train_final, y_train)
        
        # Evaluate ensemble
        metrics = ensemble.evaluate_ensemble(X_train_final, y_train)
        
        # Make predictions
        predictions = ensemble.predict(X_test_final)
        
        # Create submission
        ensemble.create_submission(predictions, test_df, sample_df, 'submission.csv')
        
        # Plot feature importance
        ensemble.plot_feature_importance(top_n=20)
        
        # Save model
        ensemble.save_model('shell_ai_ensemble_model.pkl')
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"🏆 Expected leaderboard score: >97 (based on CV RMSE: {metrics['cv_rmse_mean']:.4f})")
        print("📁 Files generated:")
        print("   - submission.csv (main submission file)")
        print("   - feature_importance.png (feature analysis)")
        print("   - shell_ai_ensemble_model.pkl (saved model)")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()