#!/usr/bin/env python3
"""
Shell.ai Hackathon 2025 - FAST ADVANCED SOLUTION
================================================

Fast implementation of advanced concepts:
- Property-wise modeling (separate models per target)
- Streamlined feature engineering (most impactful features)
- Efficient ensemble regression
- Quick cross-validation
- Optimized for MAE performance

Target runtime: 2-3 minutes
"""

import csv
import math
import json
import os
from datetime import datetime
import random

# Set seed for reproducibility
random.seed(42)

class FastAdvancedFeatureEngineer:
    """Fast advanced feature engineering - only the most impactful features"""
    
    def __init__(self):
        self.component_cols = []
        self.property_cols = []
        
    def fit(self, headers, data):
        """Fit the feature engineer on training data"""
        self.component_cols = [i for i, h in enumerate(headers) if 'fraction' in h.lower()]
        self.property_cols = [i for i, h in enumerate(headers) if 'Property' in h]
        
        print(f"🔧 Found {len(self.component_cols)} component columns")
        print(f"🔧 Found {len(self.property_cols)} property columns")
        
        return self
    
    def transform(self, headers, data):
        """Fast feature engineering - only essential features"""
        print("⚡ Fast advanced feature engineering...")
        
        new_data = []
        
        for row_idx, row in enumerate(data):
            if row_idx % 1000 == 0:
                print(f"   Processing row {row_idx}/{len(data)}")
            
            new_row = row.copy()
            
            # 1. Essential component statistics
            if self.component_cols:
                comp_values = [row[i] for i in self.component_cols if isinstance(row[i], (int, float))]
                if comp_values:
                    new_row.extend([
                        sum(comp_values) / len(comp_values),  # Mean
                        math.sqrt(sum((x - sum(comp_values) / len(comp_values)) ** 2 for x in comp_values) / len(comp_values)),  # Std
                        min(comp_values),  # Min
                        max(comp_values),  # Max
                        sorted(comp_values)[len(comp_values) // 2],  # Median
                        sum(comp_values)  # Total
                    ])
                else:
                    new_row.extend([0.0] * 6)
            
            # 2. Essential property statistics
            if self.property_cols:
                prop_values = [row[i] for i in self.property_cols if isinstance(row[i], (int, float))]
                if prop_values:
                    prop_mean = sum(prop_values) / len(prop_values)
                    prop_std = math.sqrt(sum((x - prop_mean) ** 2 for x in prop_values) / len(prop_values))
                    new_row.extend([
                        prop_mean,  # Mean
                        prop_std,   # Std
                        min(prop_values),  # Min
                        max(prop_values),  # Max
                        sorted(prop_values)[len(prop_values) // 2]  # Median
                    ])
                else:
                    new_row.extend([0.0] * 5)
            
            # 3. KEY weighted features (most impactful)
            weighted_features = self._create_key_weighted_features(headers, row)
            new_row.extend(weighted_features)
            
            # 4. TOP component interactions (limited)
            interaction_features = self._create_top_interactions(headers, row)
            new_row.extend(interaction_features)
            
            # 5. Essential polynomial features (top 3 components only)
            poly_features = self._create_essential_polynomials(headers, row)
            new_row.extend(poly_features)
            
            new_data.append(new_row)
        
        print(f"✅ Features: {len(headers)} → {len(new_data[0]) if new_data else 0}")
        return new_data
    
    def _create_key_weighted_features(self, headers, row):
        """Create only the most impactful weighted features"""
        weighted_features = []
        
        # Only top 3 components for speed
        for comp in range(1, 4):
            comp_frac = 0.0
            comp_props = []
            
            # Find component fraction
            for i, h in enumerate(headers):
                if f'Component{comp}_fraction' in h and isinstance(row[i], (int, float)):
                    comp_frac = row[i]
                    break
            
            # Find component properties (limit to first 5)
            for i, h in enumerate(headers):
                if f'Component{comp}' in h and 'Property' in h and isinstance(row[i], (int, float)):
                    comp_props.append(row[i])
                    if len(comp_props) >= 5:  # Limit for speed
                        break
            
            if comp_props:
                avg_prop = sum(comp_props) / len(comp_props)
                weighted_features.extend([
                    comp_frac * avg_prop,  # Weighted average
                    avg_prop / (comp_frac + 1e-8),  # Property intensity
                ])
            else:
                weighted_features.extend([0.0, 0.0])
        
        return weighted_features
    
    def _create_top_interactions(self, headers, row):
        """Create only top component interactions"""
        interaction_features = []
        
        # Get top 3 component values
        comp_values = []
        for i in self.component_cols[:3]:  # Top 3 only
            if i < len(row) and isinstance(row[i], (int, float)):
                comp_values.append(row[i])
            else:
                comp_values.append(0.0)
        
        # Only most important pairwise interactions
        if len(comp_values) >= 2:
            for i in range(len(comp_values)):
                for j in range(i + 1, len(comp_values)):
                    interaction_features.extend([
                        comp_values[i] * comp_values[j],  # Product
                        abs(comp_values[i] - comp_values[j]),  # Difference
                    ])
        
        return interaction_features
    
    def _create_essential_polynomials(self, headers, row):
        """Create essential polynomial features"""
        poly_features = []
        
        # Only top 3 components
        for i in self.component_cols[:3]:
            if i < len(row) and isinstance(row[i], (int, float)):
                val = row[i]
                poly_features.extend([
                    val ** 2,  # Square
                    math.sqrt(abs(val)),  # Square root
                ])
            else:
                poly_features.extend([0.0, 0.0])
        
        return poly_features


class FastPropertyWiseRegressor:
    """Fast property-wise regression with streamlined ensemble"""
    
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        
    def fast_ridge_regression(self, X, y, alpha=1.0):
        """Fast ridge regression implementation"""
        if not X or not X[0] or not y:
            return [0.0] * (len(X[0]) + 1 if X and X[0] else 1)
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Add bias term
        X_with_bias = [[1.0] + row for row in X]
        n_features += 1
        
        # X^T * X with regularization
        XTX = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                for k in range(n_samples):
                    XTX[i][j] += X_with_bias[k][i] * X_with_bias[k][j]
            XTX[i][i] += alpha  # Regularization
        
        # X^T * y
        XTy = [0.0] * n_features
        for i in range(n_features):
            for k in range(n_samples):
                XTy[i] += X_with_bias[k][i] * y[k]
        
        # Fast solve using Gaussian elimination
        return self._fast_solve(XTX, XTy)
    
    def fast_ensemble_regression(self, X, y):
        """Fast ensemble with 3 models only"""
        # Model 1: Light regularization
        weights_light = self.fast_ridge_regression(X, y, alpha=0.1)
        
        # Model 2: Medium regularization
        weights_medium = self.fast_ridge_regression(X, y, alpha=1.0)
        
        # Model 3: Strong regularization
        weights_strong = self.fast_ridge_regression(X, y, alpha=10.0)
        
        # Simple ensemble: 20% light, 60% medium, 20% strong
        n_features = len(weights_light)
        final_weights = [0.0] * n_features
        
        for i in range(n_features):
            final_weights[i] = (0.2 * weights_light[i] + 
                              0.6 * weights_medium[i] + 
                              0.2 * weights_strong[i])
        
        return final_weights
    
    def fast_cross_validate(self, X, y, k=3):
        """Fast 3-fold cross-validation"""
        n_samples = len(X)
        fold_size = n_samples // k
        
        mae_scores = []
        
        for fold in range(k):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k - 1 else n_samples
            
            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            
            X_train_fold = X[:start_idx] + X[end_idx:]
            y_train_fold = y[:start_idx] + y[end_idx:]
            
            # Train model
            weights = self.fast_ensemble_regression(X_train_fold, y_train_fold)
            
            # Predict and calculate MAE
            mae = 0.0
            for i, x_val in enumerate(X_val):
                pred = sum(weights[j] * ([1.0] + x_val)[j] for j in range(min(len(weights), len(x_val) + 1)))
                mae += abs(pred - y_val[i])
            
            mae /= len(X_val)
            mae_scores.append(mae)
        
        return mae_scores
    
    def train_property_model(self, X, y, property_name):
        """Train fast model for a specific property"""
        print(f"⚡ Training fast model for {property_name}")
        
        # Fast 3-fold cross-validation
        cv_scores = self.fast_cross_validate(X, y)
        
        # Train final model on all data
        weights = self.fast_ensemble_regression(X, y)
        
        # Store results
        self.models[property_name] = weights
        self.cv_scores[property_name] = {
            'mae_mean': sum(cv_scores) / len(cv_scores),
            'mae_std': math.sqrt(sum((s - sum(cv_scores) / len(cv_scores)) ** 2 for s in cv_scores) / len(cv_scores)),
            'scores': cv_scores
        }
        
        print(f"   {property_name} - CV MAE: {self.cv_scores[property_name]['mae_mean']:.4f} ± {self.cv_scores[property_name]['mae_std']:.4f}")
        
        return weights
    
    def predict_property(self, X, property_name):
        """Predict for a specific property"""
        if property_name not in self.models:
            raise ValueError(f"Model for {property_name} not trained")
        
        weights = self.models[property_name]
        predictions = []
        
        for x in X:
            pred = sum(weights[j] * ([1.0] + x)[j] for j in range(min(len(weights), len(x) + 1)))
            predictions.append(pred)
        
        return predictions
    
    def _fast_solve(self, A, b):
        """Fast linear system solver"""
        n = len(A)
        
        # Create augmented matrix
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        # Forward elimination (simplified)
        for i in range(n):
            # Simple pivoting
            if abs(aug[i][i]) < 1e-10:
                aug[i][i] = 1e-6
            
            # Eliminate
            for k in range(i + 1, n):
                if abs(aug[i][i]) > 1e-12:
                    factor = aug[k][i] / aug[i][i]
                    for j in range(i, n + 1):
                        aug[k][j] -= factor * aug[i][j]
        
        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            if abs(aug[i][i]) > 1e-12:
                x[i] /= aug[i][i]
        
        return x


def load_csv(filepath):
    """Load CSV file"""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = []
        for row in reader:
            numeric_row = []
            for val in row:
                try:
                    if val == '' or val.lower() == 'nan':
                        numeric_row.append(0.0)
                    else:
                        numeric_row.append(float(val))
                except ValueError:
                    numeric_row.append(val)  # Keep as string (like ID)
            data.append(numeric_row)
    return headers, data


def fast_robust_scale(X_train, X_test):
    """Fast robust scaling using simple statistics"""
    if not X_train or not X_train[0]:
        return X_train, X_test
    
    n_features = len(X_train[0])
    means = []
    stds = []
    
    for j in range(n_features):
        feature_values = [row[j] for row in X_train if isinstance(row[j], (int, float))]
        if feature_values:
            mean_val = sum(feature_values) / len(feature_values)
            variance = sum((x - mean_val) ** 2 for x in feature_values) / len(feature_values)
            std_val = math.sqrt(variance)
            
            means.append(mean_val)
            stds.append(max(std_val, 1e-8))
        else:
            means.append(0.0)
            stds.append(1.0)
    
    # Scale data
    def scale_data(data):
        scaled_data = []
        for row in data:
            scaled_row = []
            for j, val in enumerate(row):
                if isinstance(val, (int, float)):
                    scaled_val = (val - means[j]) / stds[j]
                    # Light clipping
                    scaled_val = max(-3, min(3, scaled_val))
                    scaled_row.append(scaled_val)
                else:
                    scaled_row.append(0.0)
            scaled_data.append(scaled_row)
        return scaled_data
    
    return scale_data(X_train), scale_data(X_test)


def create_submission(predictions, test_data, test_headers, target_names, filename='submission.csv'):
    """Create submission file"""
    print("📝 Creating submission...")
    
    # Find ID column
    id_col_idx = test_headers.index('ID') if 'ID' in test_headers else None
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['ID'] + target_names)
        
        # Write data
        for i, pred_row in enumerate(predictions):
            submission_row = []
            
            # Add ID
            if id_col_idx is not None and i < len(test_data):
                submission_row.append(test_data[i][id_col_idx])
            else:
                submission_row.append(i + 1)
            
            # Add predictions
            submission_row.extend(pred_row)
            writer.writerow(submission_row)
    
    print(f"✅ Submission saved to {filename}")
    
    # Log prediction statistics
    for i, target_name in enumerate(target_names):
        target_preds = [row[i] for row in predictions]
        mean_pred = sum(target_preds) / len(target_preds)
        std_pred = math.sqrt(sum((x - mean_pred) ** 2 for x in target_preds) / len(target_preds))
        print(f"   {target_name}: mean={mean_pred:.3f}, std={std_pred:.3f}")


def main():
    """Main pipeline execution"""
    print("⚡ FAST ADVANCED SHELL.AI PIPELINE")
    print("=" * 50)
    print("🎯 Property-wise modeling (optimized)")
    print("🔧 Advanced features (streamlined)")
    print("📊 Fast ensemble regression")
    print("⏱️ Target runtime: 2-3 minutes")
    print()
    
    start_time = datetime.now()
    
    try:
        # Load data
        print("📂 Loading data...")
        train_headers, train_data = load_csv('train.csv')
        test_headers, test_data = load_csv('test.csv')
        
        print(f"   Train: {len(train_data)} × {len(train_headers)}")
        print(f"   Test:  {len(test_data)} × {len(test_headers)}")
        
        # Identify targets and features
        target_indices = [i for i, h in enumerate(train_headers) if 'BlendProperty' in h]
        target_names = [train_headers[i] for i in target_indices]
        feature_indices = [i for i, h in enumerate(train_headers) 
                          if h not in ['ID'] and i not in target_indices]
        
        print(f"🎯 Found {len(target_names)} target properties")
        print(f"📊 Found {len(feature_indices)} base features")
        
        # Extract features and targets
        X_train_raw = [[row[i] for i in feature_indices] for row in train_data]
        y_train = {target_names[i]: [row[target_indices[i]] for row in train_data] 
                  for i in range(len(target_names))}
        X_test_raw = [[row[i] for i in feature_indices] for row in test_data]
        
        # Fast advanced feature engineering
        print("\n⚡ Fast Advanced Feature Engineering")
        feature_engineer = FastAdvancedFeatureEngineer()
        feature_engineer.fit([train_headers[i] for i in feature_indices], X_train_raw)
        
        X_train_engineered = feature_engineer.transform([train_headers[i] for i in feature_indices], X_train_raw)
        X_test_engineered = feature_engineer.transform([train_headers[i] for i in feature_indices], X_test_raw)
        
        # Fast scaling
        print("📊 Fast robust scaling...")
        X_train_scaled, X_test_scaled = fast_robust_scale(X_train_engineered, X_test_engineered)
        
        # Fast property-wise modeling
        print("\n⚡ Fast Property-wise Modeling")
        regressor = FastPropertyWiseRegressor()
        
        # Train models for each property
        for target_name in target_names:
            regressor.train_property_model(X_train_scaled, y_train[target_name], target_name)
        
        # Make predictions
        print("\n🔮 Making predictions...")
        predictions = []
        for i in range(len(X_test_scaled)):
            pred_row = []
            for target_name in target_names:
                pred = regressor.predict_property([X_test_scaled[i]], target_name)[0]
                pred_row.append(pred)
            predictions.append(pred_row)
        
        # Create submission
        create_submission(predictions, test_data, test_headers, target_names)
        
        # Calculate runtime
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # Final summary
        print("\n" + "=" * 50)
        print("🎉 FAST ADVANCED PIPELINE COMPLETED!")
        print(f"📊 Features engineered: {len(X_train_scaled[0]) if X_train_scaled else 0}")
        print(f"🎯 Properties modeled: {len(target_names)}")
        
        # Calculate average MAE
        avg_mae = sum(scores['mae_mean'] for scores in regressor.cv_scores.values()) / len(regressor.cv_scores)
        print(f"📈 Average CV MAE: {avg_mae:.4f}")
        
        print(f"⏱️ Runtime: {runtime:.1f} seconds")
        print(f"📁 Submission: submission.csv")
        print("🏆 Expected: Significantly improved performance!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()