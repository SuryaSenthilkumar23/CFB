#!/usr/bin/env python3
"""
Shell.ai Hackathon 2025 - Multi-Target Stacked Ensemble Solution
================================================================

High-performance stacked ensemble for fuel blend property prediction.
Predicts 10 blend properties simultaneously.
Target: Maximize private leaderboard score (goal: >97)

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any
import os
import sys

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)

class ShellAIMultiTargetEnsemble:
    """
    Multi-Target Stacked Ensemble for Shell.ai Hackathon 2025
    
    Features:
    - Handles 10 target blend properties simultaneously
    - Advanced feature engineering for fuel blend prediction
    - Pure Python implementation (no external ML libraries needed)
    - Ensemble of multiple regression models
    - Cross-validation for robust predictions
    """
    
    def __init__(self, random_state: int = 42, n_folds: int = 5):
        self.random_state = random_state
        self.n_folds = n_folds
        self.feature_names = None
        self.target_names = None
        self.models = {}
        self.scalers = {}
        
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
        
        # Handle missing values
        print(f"\n🔍 Data Quality Check:")
        train_missing = train_df.isnull().sum().sum()
        test_missing = test_df.isnull().sum().sum()
        print(f"   Train missing values: {train_missing}")
        print(f"   Test missing values: {test_missing}")
        
        if train_missing > 0:
            print("⚠️  Handling missing values in training data...")
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
            
        if test_missing > 0:
            print("⚠️  Handling missing values in test data...")
            numeric_cols = test_df.select_dtypes(include=[np.number]).columns
            test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())
        
        return train_df, test_df, sample_df
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Advanced feature engineering for fuel blend prediction"""
        print("🔧 Engineering features...")
        
        def add_features(df):
            df_new = df.copy()
            
            # Component fractions
            component_cols = [col for col in df.columns if 'fraction' in col.lower()]
            print(f"   Found {len(component_cols)} component fraction columns")
            
            if len(component_cols) > 0:
                # Component statistics
                df_new['total_components'] = df_new[component_cols].sum(axis=1)
                df_new['max_component'] = df_new[component_cols].max(axis=1)
                df_new['min_component'] = df_new[component_cols].min(axis=1)
                df_new['component_range'] = df_new['max_component'] - df_new['min_component']
                df_new['component_std'] = df_new[component_cols].std(axis=1)
                df_new['component_entropy'] = -np.sum(df_new[component_cols] * np.log(df_new[component_cols] + 1e-8), axis=1)
                
                # Dominant component analysis
                df_new['dominant_component'] = df_new[component_cols].idxmax(axis=1)
                df_new['dominant_component_value'] = df_new[component_cols].max(axis=1)
                
                # Component ratios
                for i in range(len(component_cols)):
                    for j in range(i+1, len(component_cols)):
                        col1, col2 = component_cols[i], component_cols[j]
                        df_new[f'ratio_{col1}_{col2}'] = df_new[col1] / (df_new[col2] + 1e-8)
            
            # Property features by component
            property_cols = [col for col in df.columns if 'Property' in col]
            print(f"   Found {len(property_cols)} property columns")
            
            if len(property_cols) > 0:
                # Overall property statistics
                df_new['mean_all_properties'] = df_new[property_cols].mean(axis=1)
                df_new['std_all_properties'] = df_new[property_cols].std(axis=1)
                df_new['max_all_properties'] = df_new[property_cols].max(axis=1)
                df_new['min_all_properties'] = df_new[property_cols].min(axis=1)
                df_new['range_all_properties'] = df_new['max_all_properties'] - df_new['min_all_properties']
                
                # Component-wise property statistics
                for comp in range(1, 6):  # Components 1-5
                    comp_props = [col for col in property_cols if f'Component{comp}' in col]
                    if comp_props:
                        df_new[f'mean_comp{comp}_properties'] = df_new[comp_props].mean(axis=1)
                        df_new[f'std_comp{comp}_properties'] = df_new[comp_props].std(axis=1)
                        df_new[f'max_comp{comp}_properties'] = df_new[comp_props].max(axis=1)
                        df_new[f'min_comp{comp}_properties'] = df_new[comp_props].min(axis=1)
                
                # Property-wise statistics across components
                for prop in range(1, 11):  # Properties 1-10
                    prop_cols = [col for col in property_cols if f'Property{prop}' in col]
                    if prop_cols:
                        df_new[f'mean_prop{prop}_across_comps'] = df_new[prop_cols].mean(axis=1)
                        df_new[f'std_prop{prop}_across_comps'] = df_new[prop_cols].std(axis=1)
                        df_new[f'max_prop{prop}_across_comps'] = df_new[prop_cols].max(axis=1)
                        df_new[f'min_prop{prop}_across_comps'] = df_new[prop_cols].min(axis=1)
            
            # Interaction features between components and properties
            if len(component_cols) > 0 and len(property_cols) > 0:
                for comp in range(1, 6):
                    comp_frac_col = f'Component{comp}_fraction'
                    if comp_frac_col in df.columns:
                        comp_props = [col for col in property_cols if f'Component{comp}' in col]
                        for prop_col in comp_props:
                            # Weighted property by component fraction
                            df_new[f'weighted_{prop_col}'] = df_new[comp_frac_col] * df_new[prop_col]
            
            # Polynomial features for key variables
            key_cols = component_cols + [col for col in df_new.columns if 'mean_' in col][:10]
            for col in key_cols:
                if col in df_new.columns and df_new[col].std() > 1e-8:
                    df_new[f'{col}_squared'] = df_new[col] ** 2
                    df_new[f'{col}_sqrt'] = np.sqrt(np.abs(df_new[col]))
            
            return df_new
        
        # Apply feature engineering
        train_enhanced = add_features(train_df)
        test_enhanced = add_features(test_df)
        
        print(f"✅ Feature engineering complete:")
        print(f"   Original features: {train_df.shape[1]}")
        print(f"   Enhanced features: {train_enhanced.shape[1]}")
        
        return train_enhanced, test_enhanced
    
    def prepare_features_and_targets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for modeling"""
        print("📋 Preparing features and targets...")
        
        # Identify target columns (BlendProperty1-10)
        target_cols = [col for col in train_df.columns if 'BlendProperty' in col]
        if not target_cols:
            # Fallback: assume last 10 columns are targets
            target_cols = train_df.columns[-10:].tolist()
        
        self.target_names = sorted(target_cols)
        print(f"🎯 Target columns: {self.target_names}")
        
        # Prepare features (exclude ID and target columns)
        feature_cols = [col for col in train_df.columns if col not in ['ID'] + target_cols]
        self.feature_names = feature_cols
        print(f"📊 Feature columns: {len(feature_cols)}")
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_cols].values
        X_test = test_df[feature_cols].values
        
        print(f"📊 Final dataset shapes:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   X_test: {X_test.shape}")
        
        # Handle infinite and missing values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return X_train, y_train, X_test
    
    def standardize_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize features using robust scaling"""
        print("⚖️  Standardizing features...")
        
        # Calculate robust statistics (median and MAD)
        medians = np.median(X_train, axis=0)
        mads = np.median(np.abs(X_train - medians), axis=0)
        
        # Avoid division by zero
        mads = np.where(mads == 0, 1, mads)
        
        # Store scalers
        self.scalers = {'medians': medians, 'mads': mads}
        
        # Apply scaling
        X_train_scaled = (X_train - medians) / mads
        X_test_scaled = (X_test - medians) / mads
        
        return X_train_scaled, X_test_scaled
    
    def ridge_regression(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Ridge regression implementation"""
        n_features = X.shape[1]
        I = np.eye(n_features)
        
        try:
            # Ridge regression: (X^T X + alpha*I)^(-1) X^T y
            XtX = X.T @ X
            XtX_reg = XtX + alpha * I
            Xty = X.T @ y
            weights = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            weights = np.linalg.pinv(X.T @ X + alpha * I) @ X.T @ y
            
        return weights
    
    def predict_ridge(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Make predictions using ridge regression weights"""
        return X @ weights
    
    def cross_validate_alpha(self, X: np.ndarray, y: np.ndarray, alphas: List[float]) -> float:
        """Find best alpha using cross-validation"""
        n_samples = X.shape[0]
        fold_size = n_samples // self.n_folds
        best_alpha = alphas[0]
        best_score = float('inf')
        
        for alpha in alphas:
            cv_scores = []
            
            for fold in range(self.n_folds):
                # Create train/val split
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.n_folds - 1 else n_samples
                
                val_idx = list(range(val_start, val_end))
                train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
                
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                # Train and predict
                weights = self.ridge_regression(X_fold_train, y_fold_train, alpha)
                y_pred = self.predict_ridge(X_fold_val, weights)
                
                # Calculate MSE
                mse = np.mean((y_fold_val - y_pred) ** 2)
                cv_scores.append(mse)
            
            avg_score = np.mean(cv_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
        
        return best_alpha
    
    def train_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train ensemble of models for each target"""
        print("🏗️  Training multi-target ensemble...")
        
        n_targets = y_train.shape[1]
        
        # Alpha values to test
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        for target_idx in range(n_targets):
            target_name = self.target_names[target_idx]
            print(f"   Training models for {target_name}...")
            
            y_target = y_train[:, target_idx]
            
            # Find best alpha for this target
            best_alpha = self.cross_validate_alpha(X_train, y_target, alphas)
            print(f"     Best alpha: {best_alpha}")
            
            # Train multiple models with different regularization
            models = {}
            
            # Ridge with optimal alpha
            models['ridge_optimal'] = self.ridge_regression(X_train, y_target, best_alpha)
            
            # Ridge with different alphas for diversity
            models['ridge_low'] = self.ridge_regression(X_train, y_target, best_alpha * 0.1)
            models['ridge_high'] = self.ridge_regression(X_train, y_target, best_alpha * 10)
            
            # Linear regression (alpha = 0)
            models['linear'] = self.ridge_regression(X_train, y_target, 0.001)
            
            self.models[target_name] = models
        
        print("✅ Multi-target ensemble training complete!")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions for all targets"""
        print("🔮 Making multi-target predictions...")
        
        n_samples = X_test.shape[0]
        n_targets = len(self.target_names)
        predictions = np.zeros((n_samples, n_targets))
        
        for target_idx, target_name in enumerate(self.target_names):
            models = self.models[target_name]
            
            # Get predictions from all models for this target
            target_predictions = []
            for model_name, weights in models.items():
                pred = self.predict_ridge(X_test, weights)
                target_predictions.append(pred)
            
            # Ensemble prediction (simple average)
            ensemble_pred = np.mean(target_predictions, axis=0)
            predictions[:, target_idx] = ensemble_pred
        
        return predictions
    
    def evaluate_performance(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance using cross-validation"""
        print("📊 Evaluating multi-target performance...")
        
        n_samples = X_train.shape[0]
        fold_size = n_samples // self.n_folds
        
        cv_scores = {target: [] for target in self.target_names}
        
        for fold in range(self.n_folds):
            # Create train/val split
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_folds - 1 else n_samples
            
            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train models on fold
            temp_models = {}
            for target_idx, target_name in enumerate(self.target_names):
                y_target = y_fold_train[:, target_idx]
                best_alpha = self.cross_validate_alpha(X_fold_train, y_target, [0.1, 1.0, 10.0])
                temp_models[target_name] = self.ridge_regression(X_fold_train, y_target, best_alpha)
            
            # Make predictions
            for target_idx, target_name in enumerate(self.target_names):
                weights = temp_models[target_name]
                y_pred = self.predict_ridge(X_fold_val, weights)
                y_true = y_fold_val[:, target_idx]
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                cv_scores[target_name].append(rmse)
        
        # Calculate average scores
        avg_scores = {}
        for target_name in self.target_names:
            avg_scores[target_name] = np.mean(cv_scores[target_name])
        
        overall_rmse = np.mean(list(avg_scores.values()))
        avg_scores['overall'] = overall_rmse
        
        print(f"📊 Cross-Validation Results:")
        for target_name, score in avg_scores.items():
            print(f"   {target_name}: RMSE = {score:.4f}")
        
        return avg_scores
    
    def create_submission(self, predictions: np.ndarray, test_df: pd.DataFrame, sample_df: pd.DataFrame, output_path: str = 'submission.csv') -> None:
        """Create submission file"""
        print("📝 Creating submission file...")
        
        # Create submission dataframe
        submission = pd.DataFrame()
        submission['ID'] = test_df['ID']
        
        # Add predictions for each target
        for i, target_name in enumerate(self.target_names):
            submission[target_name] = predictions[:, i]
        
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
    print("🚀 Shell.ai Hackathon 2025 - Multi-Target Stacked Ensemble")
    print("=" * 70)
    
    # Initialize ensemble
    ensemble = ShellAIMultiTargetEnsemble(random_state=42, n_folds=5)
    
    try:
        # Load and preprocess data
        train_df, test_df, sample_df = ensemble.load_and_preprocess_data(
            'train.csv', 'test.csv', 'sample_solution.csv'
        )
        
        # Feature engineering
        train_enhanced, test_enhanced = ensemble.engineer_features(train_df, test_df)
        
        # Prepare features and targets
        X_train, y_train, X_test = ensemble.prepare_features_and_targets(train_enhanced, test_enhanced)
        
        # Standardize features
        X_train_scaled, X_test_scaled = ensemble.standardize_features(X_train, X_test)
        
        # Train ensemble models
        ensemble.train_ensemble_models(X_train_scaled, y_train)
        
        # Evaluate performance
        cv_scores = ensemble.evaluate_performance(X_train_scaled, y_train)
        
        # Make predictions
        predictions = ensemble.predict(X_test_scaled)
        
        # Create submission
        ensemble.create_submission(predictions, test_df, sample_df, 'submission.csv')
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"🏆 Expected leaderboard score: Based on CV RMSE: {cv_scores['overall']:.4f}")
        print("📁 Files generated:")
        print("   - submission.csv (main submission file)")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()