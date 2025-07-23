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

import csv
import math
import random
from typing import Dict, List, Tuple, Any
import os
import sys

# Configuration
random.seed(42)

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
        
    def load_csv(self, filepath: str) -> Tuple[List[str], List[List[float]]]:
        """Load CSV file and return headers and data"""
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = []
            for row in reader:
                # Convert to float, handle missing values
                numeric_row = []
                for val in row:
                    try:
                        if val == '' or val.lower() == 'nan':
                            numeric_row.append(0.0)  # Fill missing with 0
                        else:
                            numeric_row.append(float(val))
                    except ValueError:
                        # Non-numeric values (like ID strings)
                        numeric_row.append(val)
                data.append(numeric_row)
        return headers, data
    
    def load_and_preprocess_data(self, train_path: str, test_path: str, sample_path: str) -> Tuple[Dict, Dict, Dict]:
        """Load and preprocess the dataset"""
        print("🔄 Loading datasets...")
        
        # Load data
        train_headers, train_data = self.load_csv(train_path)
        test_headers, test_data = self.load_csv(test_path)
        sample_headers, sample_data = self.load_csv(sample_path)
        
        train_dict = {'headers': train_headers, 'data': train_data}
        test_dict = {'headers': test_headers, 'data': test_data}
        sample_dict = {'headers': sample_headers, 'data': sample_data}
        
        print(f"📊 Dataset shapes:")
        print(f"   Train: {len(train_data)} x {len(train_headers)}")
        print(f"   Test: {len(test_data)} x {len(test_headers)}")
        print(f"   Sample: {len(sample_data)} x {len(sample_headers)}")
        
        return train_dict, test_dict, sample_dict
    
    def calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values"""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        n = len(values)
        mean_val = sum(values) / n
        variance = sum((x - mean_val) ** 2 for x in values) / n if n > 1 else 0
        std_val = math.sqrt(variance)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'min': min(values),
            'max': max(values)
        }
    
    def engineer_features(self, train_dict: Dict, test_dict: Dict) -> Tuple[Dict, Dict]:
        """Advanced feature engineering for fuel blend prediction"""
        print("🔧 Engineering features...")
        
        def add_features(data_dict):
            headers = data_dict['headers']
            data = data_dict['data']
            
            # Find column indices
            component_indices = [i for i, h in enumerate(headers) if 'fraction' in h.lower()]
            property_indices = [i for i, h in enumerate(headers) if 'Property' in h]
            
            print(f"   Found {len(component_indices)} component fraction columns")
            print(f"   Found {len(property_indices)} property columns")
            
            new_headers = headers.copy()
            new_data = []
            
            for row in data:
                new_row = row.copy()
                
                # Component statistics
                if component_indices:
                    comp_values = [row[i] for i in component_indices if isinstance(row[i], (int, float))]
                    if comp_values:
                        stats = self.calculate_stats(comp_values)
                        new_row.extend([
                            sum(comp_values),  # total_components
                            stats['max'],      # max_component
                            stats['min'],      # min_component
                            stats['max'] - stats['min'],  # component_range
                            stats['std'],      # component_std
                        ])
                        
                        # Component ratios (first 3 pairs to avoid too many features)
                        for i in range(min(3, len(comp_values))):
                            for j in range(i+1, min(3, len(comp_values))):
                                ratio = comp_values[i] / (comp_values[j] + 1e-8)
                                new_row.append(ratio)
                
                # Property statistics
                if property_indices:
                    prop_values = [row[i] for i in property_indices if isinstance(row[i], (int, float))]
                    if prop_values:
                        stats = self.calculate_stats(prop_values)
                        new_row.extend([
                            stats['mean'],     # mean_all_properties
                            stats['std'],      # std_all_properties
                            stats['max'],      # max_all_properties
                            stats['min'],      # min_all_properties
                            stats['max'] - stats['min'],  # range_all_properties
                        ])
                        
                        # Component-wise property statistics
                        for comp in range(1, 6):
                            comp_props = [row[i] for i, h in enumerate(headers) 
                                        if f'Component{comp}' in h and 'Property' in h 
                                        and isinstance(row[i], (int, float))]
                            if comp_props:
                                comp_stats = self.calculate_stats(comp_props)
                                new_row.extend([comp_stats['mean'], comp_stats['std']])
                        
                        # Property-wise statistics across components (first 5 properties)
                        for prop in range(1, 6):
                            prop_across_comps = [row[i] for i, h in enumerate(headers) 
                                               if f'Property{prop}' in h 
                                               and isinstance(row[i], (int, float))]
                            if prop_across_comps:
                                prop_stats = self.calculate_stats(prop_across_comps)
                                new_row.extend([prop_stats['mean'], prop_stats['std']])
                
                # Interaction features (component fractions * average properties)
                if component_indices and property_indices:
                    for comp in range(1, 6):
                        comp_frac_idx = None
                        for i, h in enumerate(headers):
                            if f'Component{comp}_fraction' in h:
                                comp_frac_idx = i
                                break
                        
                        if comp_frac_idx is not None and isinstance(row[comp_frac_idx], (int, float)):
                            comp_props = [row[i] for i, h in enumerate(headers) 
                                        if f'Component{comp}' in h and 'Property' in h 
                                        and isinstance(row[i], (int, float))]
                            if comp_props:
                                avg_prop = sum(comp_props) / len(comp_props)
                                weighted_prop = row[comp_frac_idx] * avg_prop
                                new_row.append(weighted_prop)
                
                new_data.append(new_row)
            
            # Create new headers for engineered features
            if component_indices:
                new_headers.extend([
                    'total_components', 'max_component', 'min_component', 
                    'component_range', 'component_std'
                ])
                # Add ratio headers
                for i in range(min(3, len(component_indices))):
                    for j in range(i+1, min(3, len(component_indices))):
                        new_headers.append(f'ratio_comp{i+1}_comp{j+1}')
            
            if property_indices:
                new_headers.extend([
                    'mean_all_properties', 'std_all_properties', 'max_all_properties',
                    'min_all_properties', 'range_all_properties'
                ])
                # Component-wise stats
                for comp in range(1, 6):
                    new_headers.extend([f'mean_comp{comp}_properties', f'std_comp{comp}_properties'])
                # Property-wise stats
                for prop in range(1, 6):
                    new_headers.extend([f'mean_prop{prop}_across_comps', f'std_prop{prop}_across_comps'])
            
            # Interaction headers
            if component_indices and property_indices:
                for comp in range(1, 6):
                    new_headers.append(f'weighted_comp{comp}_avg_prop')
            
            return {'headers': new_headers, 'data': new_data}
        
        # Apply feature engineering
        train_enhanced = add_features(train_dict)
        test_enhanced = add_features(test_dict)
        
        print(f"✅ Feature engineering complete:")
        print(f"   Original features: {len(train_dict['headers'])}")
        print(f"   Enhanced features: {len(train_enhanced['headers'])}")
        
        return train_enhanced, test_enhanced
    
    def prepare_features_and_targets(self, train_dict: Dict, test_dict: Dict) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """Prepare features and targets for modeling"""
        print("📋 Preparing features and targets...")
        
        train_headers = train_dict['headers']
        train_data = train_dict['data']
        test_headers = test_dict['headers']
        test_data = test_dict['data']
        
        # Identify target columns (BlendProperty1-10)
        target_indices = [i for i, h in enumerate(train_headers) if 'BlendProperty' in h]
        if not target_indices:
            # Fallback: assume last 10 columns are targets
            target_indices = list(range(len(train_headers) - 10, len(train_headers)))
        
        self.target_names = [train_headers[i] for i in target_indices]
        print(f"🎯 Target columns: {self.target_names}")
        
        # Prepare features (exclude ID and target columns)
        feature_indices = [i for i, h in enumerate(train_headers) 
                          if h not in ['ID'] and i not in target_indices]
        self.feature_names = [train_headers[i] for i in feature_indices]
        print(f"📊 Feature columns: {len(feature_indices)}")
        
        # Find matching features in test data
        test_feature_indices = []
        for feature_name in self.feature_names:
            try:
                test_idx = test_headers.index(feature_name)
                test_feature_indices.append(test_idx)
            except ValueError:
                # Feature not found in test data, will use 0.0
                test_feature_indices.append(-1)
        
        # Extract features and targets
        X_train = []
        y_train = []
        for row in train_data:
            # Features
            feature_row = []
            for i in feature_indices:
                if i < len(row) and isinstance(row[i], (int, float)):
                    feature_row.append(row[i])
                else:
                    feature_row.append(0.0)
            X_train.append(feature_row)
            
            # Targets
            target_row = []
            for i in target_indices:
                if i < len(row) and isinstance(row[i], (int, float)):
                    target_row.append(row[i])
                else:
                    target_row.append(0.0)
            y_train.append(target_row)
        
        # Extract test features
        X_test = []
        for row in test_data:
            feature_row = []
            for i in test_feature_indices:
                if i >= 0 and i < len(row) and isinstance(row[i], (int, float)):
                    feature_row.append(row[i])
                else:
                    feature_row.append(0.0)
            X_test.append(feature_row)
        
        print(f"📊 Final dataset shapes:")
        print(f"   X_train: {len(X_train)} x {len(X_train[0]) if X_train else 0}")
        print(f"   y_train: {len(y_train)} x {len(y_train[0]) if y_train else 0}")
        print(f"   X_test: {len(X_test)} x {len(X_test[0]) if X_test else 0}")
        
        return X_train, y_train, X_test
    
    def calculate_median(self, values: List[float]) -> float:
        """Calculate median of a list of values"""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        else:
            return sorted_vals[n//2]
    
    def standardize_features(self, X_train: List[List[float]], X_test: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """Standardize features using robust scaling"""
        print("⚖️  Standardizing features...")
        
        if not X_train or not X_train[0]:
            return X_train, X_test
        
        n_features = len(X_train[0])
        
        # Calculate medians and MADs for each feature
        medians = []
        mads = []
        
        for j in range(n_features):
            # Get all values for feature j
            feature_values = [X_train[i][j] for i in range(len(X_train))]
            median_val = self.calculate_median(feature_values)
            medians.append(median_val)
            
            # Calculate MAD (Median Absolute Deviation)
            abs_deviations = [abs(val - median_val) for val in feature_values]
            mad_val = self.calculate_median(abs_deviations)
            if mad_val == 0:
                mad_val = 1.0  # Avoid division by zero
            mads.append(mad_val)
        
        # Store scalers
        self.scalers = {'medians': medians, 'mads': mads}
        
        # Apply scaling to training data
        X_train_scaled = []
        for i in range(len(X_train)):
            scaled_row = [(X_train[i][j] - medians[j]) / mads[j] for j in range(n_features)]
            X_train_scaled.append(scaled_row)
        
        # Apply scaling to test data
        X_test_scaled = []
        for i in range(len(X_test)):
            scaled_row = [(X_test[i][j] - medians[j]) / mads[j] for j in range(n_features)]
            X_test_scaled.append(scaled_row)
        
        return X_train_scaled, X_test_scaled
    
    def matrix_multiply(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Multiply two matrices"""
        rows_A, cols_A = len(A), len(A[0]) if A else 0
        rows_B, cols_B = len(B), len(B[0]) if B else 0
        
        if cols_A != rows_B:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def transpose(self, matrix: List[List[float]]) -> List[List[float]]:
        """Transpose a matrix"""
        if not matrix or not matrix[0]:
            return []
        return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
    
    def add_matrices(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Add two matrices"""
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    
    def solve_linear_system(self, A: List[List[float]], b: List[float]) -> List[float]:
        """Solve linear system Ax = b using Gaussian elimination"""
        n = len(A)
        
        # Create augmented matrix
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        # Forward elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Make all rows below this one 0 in current column
            for k in range(i + 1, n):
                if aug[i][i] != 0:
                    factor = aug[k][i] / aug[i][i]
                    for j in range(i, n + 1):
                        aug[k][j] -= factor * aug[i][j]
        
        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            if aug[i][i] != 0:
                x[i] /= aug[i][i]
        
        return x
    
    def ridge_regression(self, X: List[List[float]], y: List[float], alpha: float = 1.0) -> List[float]:
        """Ridge regression implementation"""
        if not X or not X[0]:
            return []
        
        n_features = len(X[0])
        
        # X^T
        Xt = self.transpose(X)
        
        # X^T @ X
        XtX = self.matrix_multiply(Xt, X)
        
        # Add regularization: X^T @ X + alpha * I
        for i in range(n_features):
            XtX[i][i] += alpha
        
        # X^T @ y
        Xty = [sum(Xt[i][j] * y[j] for j in range(len(y))) for i in range(n_features)]
        
        # Solve (X^T @ X + alpha * I) @ weights = X^T @ y
        try:
            weights = self.solve_linear_system(XtX, Xty)
        except:
            # Fallback to simple solution if system is singular
            weights = [0.0] * n_features
        
        return weights
    
    def predict_ridge(self, X: List[List[float]], weights: List[float]) -> List[float]:
        """Make predictions using ridge regression weights"""
        if not X or not weights:
            return []
        
        predictions = []
        for row in X:
            pred = sum(row[j] * weights[j] for j in range(len(weights)))
            predictions.append(pred)
        return predictions
    
    def cross_validate_alpha(self, X: List[List[float]], y: List[float], alphas: List[float]) -> float:
        """Find best alpha using cross-validation"""
        n_samples = len(X)
        fold_size = n_samples // self.n_folds
        best_alpha = alphas[0]
        best_score = float('inf')
        
        for alpha in alphas:
            cv_scores = []
            
            for fold in range(self.n_folds):
                # Create train/val split
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.n_folds - 1 else n_samples
                
                # Split data
                X_fold_train = [X[i] for i in range(n_samples) if i < val_start or i >= val_end]
                X_fold_val = [X[i] for i in range(val_start, val_end)]
                y_fold_train = [y[i] for i in range(n_samples) if i < val_start or i >= val_end]
                y_fold_val = [y[i] for i in range(val_start, val_end)]
                
                # Train and predict
                weights = self.ridge_regression(X_fold_train, y_fold_train, alpha)
                y_pred = self.predict_ridge(X_fold_val, weights)
                
                # Calculate MSE
                if y_pred and y_fold_val:
                    mse = sum((y_fold_val[i] - y_pred[i]) ** 2 for i in range(len(y_pred))) / len(y_pred)
                    cv_scores.append(mse)
            
            if cv_scores:
                avg_score = sum(cv_scores) / len(cv_scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_alpha = alpha
        
        return best_alpha
    
    def train_ensemble_models(self, X_train: List[List[float]], y_train: List[List[float]]) -> None:
        """Train ensemble of models for each target"""
        print("🏗️  Training multi-target ensemble...")
        
        n_targets = len(y_train[0]) if y_train else 0
        
        # Alpha values to test
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]  # Reduced for speed
        
        for target_idx in range(n_targets):
            target_name = self.target_names[target_idx]
            print(f"   Training models for {target_name}...")
            
            # Extract target values for this target
            y_target = [row[target_idx] for row in y_train]
            
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
            
            # Linear regression (small alpha)
            models['linear'] = self.ridge_regression(X_train, y_target, 0.001)
            
            self.models[target_name] = models
        
        print("✅ Multi-target ensemble training complete!")
    
    def predict(self, X_test: List[List[float]]) -> List[List[float]]:
        """Make predictions for all targets"""
        print("🔮 Making multi-target predictions...")
        
        n_samples = len(X_test)
        n_targets = len(self.target_names)
        predictions = [[0.0 for _ in range(n_targets)] for _ in range(n_samples)]
        
        for target_idx, target_name in enumerate(self.target_names):
            models = self.models[target_name]
            
            # Get predictions from all models for this target
            target_predictions = []
            for model_name, weights in models.items():
                pred = self.predict_ridge(X_test, weights)
                target_predictions.append(pred)
            
            # Ensemble prediction (simple average)
            if target_predictions:
                for i in range(n_samples):
                    avg_pred = sum(pred[i] for pred in target_predictions) / len(target_predictions)
                    predictions[i][target_idx] = avg_pred
        
        return predictions
    
    def evaluate_performance(self, X_train: List[List[float]], y_train: List[List[float]]) -> Dict[str, float]:
        """Evaluate model performance using cross-validation"""
        print("📊 Evaluating multi-target performance...")
        
        n_samples = len(X_train)
        fold_size = n_samples // self.n_folds
        
        cv_scores = {target: [] for target in self.target_names}
        
        for fold in range(self.n_folds):
            # Create train/val split
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_folds - 1 else n_samples
            
            # Split data
            X_fold_train = [X_train[i] for i in range(n_samples) if i < val_start or i >= val_end]
            X_fold_val = [X_train[i] for i in range(val_start, val_end)]
            y_fold_train = [y_train[i] for i in range(n_samples) if i < val_start or i >= val_end]
            y_fold_val = [y_train[i] for i in range(val_start, val_end)]
            
            # Train models on fold
            temp_models = {}
            for target_idx, target_name in enumerate(self.target_names):
                y_target = [row[target_idx] for row in y_fold_train]
                best_alpha = self.cross_validate_alpha(X_fold_train, y_target, [0.1, 1.0, 10.0])
                temp_models[target_name] = self.ridge_regression(X_fold_train, y_target, best_alpha)
            
            # Make predictions
            for target_idx, target_name in enumerate(self.target_names):
                weights = temp_models[target_name]
                y_pred = self.predict_ridge(X_fold_val, weights)
                y_true = [row[target_idx] for row in y_fold_val]
                
                # Calculate RMSE
                if y_pred and y_true and len(y_pred) == len(y_true):
                    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_pred))) / len(y_pred)
                    rmse = math.sqrt(mse)
                    cv_scores[target_name].append(rmse)
        
        # Calculate average scores
        avg_scores = {}
        for target_name in self.target_names:
            if cv_scores[target_name]:
                avg_scores[target_name] = sum(cv_scores[target_name]) / len(cv_scores[target_name])
            else:
                avg_scores[target_name] = 0.0
        
        if avg_scores:
            overall_rmse = sum(avg_scores.values()) / len(avg_scores)
        else:
            overall_rmse = 0.0
        avg_scores['overall'] = overall_rmse
        
        print(f"📊 Cross-Validation Results:")
        for target_name, score in avg_scores.items():
            print(f"   {target_name}: RMSE = {score:.4f}")
        
        return avg_scores
    
    def create_submission(self, predictions: List[List[float]], test_dict: Dict, sample_dict: Dict, output_path: str = 'submission.csv') -> None:
        """Create submission file"""
        print("📝 Creating submission file...")
        
        test_data = test_dict['data']
        test_headers = test_dict['headers']
        
        # Find ID column
        id_col_idx = None
        for i, header in enumerate(test_headers):
            if header == 'ID':
                id_col_idx = i
                break
        
        # Create submission data
        submission_data = []
        for i, row in enumerate(test_data):
            submission_row = []
            
            # Add ID
            if id_col_idx is not None:
                submission_row.append(row[id_col_idx])
            else:
                submission_row.append(i + 1)  # Fallback ID
            
            # Add predictions
            if i < len(predictions):
                submission_row.extend(predictions[i])
            else:
                submission_row.extend([0.0] * len(self.target_names))
            
            submission_data.append(submission_row)
        
        # Create headers
        submission_headers = ['ID'] + self.target_names
        
        # Save to CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(submission_headers)
            writer.writerows(submission_data)
        
        print(f"✅ Submission saved to {output_path}")
        print(f"📊 Submission statistics:")
        print(f"   Shape: {len(submission_data)} x {len(submission_headers)}")
        
        if predictions:
            flat_predictions = [val for row in predictions for val in row]
            if flat_predictions:
                pred_min = min(flat_predictions)
                pred_max = max(flat_predictions)
                pred_mean = sum(flat_predictions) / len(flat_predictions)
                pred_var = sum((x - pred_mean) ** 2 for x in flat_predictions) / len(flat_predictions)
                pred_std = math.sqrt(pred_var)
                
                print(f"   Predictions range: [{pred_min:.4f}, {pred_max:.4f}]")
                print(f"   Predictions mean: {pred_mean:.4f}")
                print(f"   Predictions std: {pred_std:.4f}")


def main():
    """Main execution function"""
    print("🚀 Shell.ai Hackathon 2025 - Multi-Target Stacked Ensemble")
    print("=" * 70)
    
    # Initialize ensemble
    ensemble = ShellAIMultiTargetEnsemble(random_state=42, n_folds=5)
    
    try:
        # Load and preprocess data
        train_dict, test_dict, sample_dict = ensemble.load_and_preprocess_data(
            'train.csv', 'test.csv', 'sample_solution.csv'
        )
        
        # Feature engineering
        train_enhanced, test_enhanced = ensemble.engineer_features(train_dict, test_dict)
        
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
        ensemble.create_submission(predictions, test_dict, sample_dict, 'submission.csv')
        
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