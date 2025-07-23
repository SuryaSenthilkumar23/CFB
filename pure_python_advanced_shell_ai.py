#!/usr/bin/env python3
"""
Shell.ai Hackathon 2025 - PURE PYTHON ADVANCED SOLUTION
=======================================================

Advanced solution implemented in pure Python with:
- Property-wise modeling (separate models per target)
- Advanced feature engineering with interactions
- Multiple regression models with ensemble
- Cross-validation and performance logging
- Optimized for MAE/MAPE metrics
"""

import csv
import math
import json
import os
from datetime import datetime
import random

# Set seed for reproducibility
random.seed(42)

class PurePythonAdvancedFeatureEngineer:
    """Advanced feature engineering using pure Python"""
    
    def __init__(self):
        self.component_cols = []
        self.property_cols = []
        self.feature_names = []
        
    def fit(self, headers, data):
        """Fit the feature engineer on training data"""
        self.component_cols = [i for i, h in enumerate(headers) if 'fraction' in h.lower()]
        self.property_cols = [i for i, h in enumerate(headers) if 'Property' in h]
        
        print(f"🔧 Found {len(self.component_cols)} component columns")
        print(f"🔧 Found {len(self.property_cols)} property columns")
        
        return self
    
    def transform(self, headers, data):
        """Transform data with advanced feature engineering"""
        print("🚀 Advanced feature engineering...")
        
        new_data = []
        
        for row_idx, row in enumerate(data):
            if row_idx % 500 == 0:
                print(f"   Processing row {row_idx}/{len(data)}")
            
            new_row = row.copy()
            
            # 1. Component statistics (enhanced)
            if self.component_cols:
                comp_values = [row[i] for i in self.component_cols if isinstance(row[i], (int, float))]
                if comp_values:
                    comp_stats = self._calculate_advanced_stats(comp_values)
                    new_row.extend([
                        comp_stats['mean'], comp_stats['std'], comp_stats['min'], comp_stats['max'],
                        comp_stats['median'], comp_stats['range'], comp_stats['cv'],
                        comp_stats['skew'], comp_stats['kurt'], sum(comp_values)
                    ])
                else:
                    new_row.extend([0.0] * 10)
            
            # 2. Property statistics (enhanced)
            if self.property_cols:
                prop_values = [row[i] for i in self.property_cols if isinstance(row[i], (int, float))]
                if prop_values:
                    prop_stats = self._calculate_advanced_stats(prop_values)
                    new_row.extend([
                        prop_stats['mean'], prop_stats['std'], prop_stats['min'], prop_stats['max'],
                        prop_stats['median'], prop_stats['range'], prop_stats['cv'],
                        prop_stats['skew'], prop_stats['kurt']
                    ])
                else:
                    new_row.extend([0.0] * 9)
            
            # 3. Weighted component-property features
            weighted_features = self._create_weighted_features(headers, row)
            new_row.extend(weighted_features)
            
            # 4. Component interactions
            interaction_features = self._create_interaction_features(headers, row)
            new_row.extend(interaction_features)
            
            # 5. Polynomial features
            poly_features = self._create_polynomial_features(headers, row)
            new_row.extend(poly_features)
            
            # 6. Transformed features
            transform_features = self._create_transformed_features(headers, row)
            new_row.extend(transform_features)
            
            # 7. Cross-component diversity features
            diversity_features = self._create_diversity_features(headers, row)
            new_row.extend(diversity_features)
            
            new_data.append(new_row)
        
        # Generate feature names
        self._generate_feature_names(headers)
        
        print(f"✅ Features: {len(headers)} → {len(new_data[0]) if new_data else 0}")
        return new_data
    
    def _calculate_advanced_stats(self, values):
        """Calculate advanced statistics"""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'range': 0, 'cv': 0, 'skew': 0, 'kurt': 0}
        
        n = len(values)
        mean_val = sum(values) / n
        
        # Standard deviation
        variance = sum((x - mean_val) ** 2 for x in values) / n if n > 1 else 0
        std_val = math.sqrt(variance)
        
        # Sorted values for percentiles
        sorted_vals = sorted(values)
        median = sorted_vals[n // 2]
        
        # Range and CV
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        cv = std_val / (abs(mean_val) + 1e-8)
        
        # Skewness and kurtosis (simplified)
        if std_val > 0:
            skew = sum((x - mean_val) ** 3 for x in values) / (n * std_val ** 3)
            kurt = sum((x - mean_val) ** 4 for x in values) / (n * std_val ** 4) - 3
        else:
            skew = 0
            kurt = 0
        
        return {
            'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val,
            'median': median, 'range': range_val, 'cv': cv, 'skew': skew, 'kurt': kurt
        }
    
    def _create_weighted_features(self, headers, row):
        """Create weighted component-property features"""
        weighted_features = []
        
        # Map components to their properties
        for comp in range(1, 6):
            comp_frac = 0.0
            comp_props = []
            
            # Find component fraction
            for i, h in enumerate(headers):
                if f'Component{comp}_fraction' in h and isinstance(row[i], (int, float)):
                    comp_frac = row[i]
                    break
            
            # Find component properties
            for i, h in enumerate(headers):
                if f'Component{comp}' in h and 'Property' in h and isinstance(row[i], (int, float)):
                    comp_props.append(row[i])
            
            if comp_props:
                # Weighted features
                avg_prop = sum(comp_props) / len(comp_props)
                max_prop = max(comp_props)
                min_prop = min(comp_props)
                
                weighted_features.extend([
                    comp_frac * avg_prop,  # Weighted average
                    comp_frac * max_prop,  # Weighted max
                    comp_frac * min_prop,  # Weighted min
                    avg_prop / (comp_frac + 1e-8),  # Property intensity
                    comp_frac * sum(comp_props),  # Weighted sum
                ])
            else:
                weighted_features.extend([0.0] * 5)
        
        return weighted_features
    
    def _create_interaction_features(self, headers, row):
        """Create component interaction features"""
        interaction_features = []
        
        # Component × Component interactions
        comp_values = []
        for i in self.component_cols:
            if i < len(row) and isinstance(row[i], (int, float)):
                comp_values.append(row[i])
            else:
                comp_values.append(0.0)
        
        # Pairwise interactions
        for i in range(len(comp_values)):
            for j in range(i + 1, len(comp_values)):
                interaction_features.extend([
                    comp_values[i] * comp_values[j],  # Product
                    abs(comp_values[i] - comp_values[j]),  # Absolute difference
                    comp_values[i] / (comp_values[j] + 1e-8),  # Ratio
                    (comp_values[i] + comp_values[j]) / 2,  # Average
                ])
        
        # Component × Property interactions (limited)
        if len(comp_values) >= 2 and len(self.property_cols) >= 2:
            top_props = []
            for i in self.property_cols[:5]:  # Top 5 properties
                if i < len(row) and isinstance(row[i], (int, float)):
                    top_props.append(row[i])
                else:
                    top_props.append(0.0)
            
            # Cross interactions
            for comp_val in comp_values[:3]:  # Top 3 components
                for prop_val in top_props[:3]:  # Top 3 properties
                    interaction_features.append(comp_val * prop_val)
        
        return interaction_features
    
    def _create_polynomial_features(self, headers, row):
        """Create polynomial features"""
        poly_features = []
        
        # Polynomial features for components
        for i in self.component_cols[:5]:  # Top 5 components
            if i < len(row) and isinstance(row[i], (int, float)):
                val = row[i]
                poly_features.extend([
                    val ** 2,  # Square
                    val ** 3,  # Cube
                    math.sqrt(abs(val)),  # Square root
                ])
            else:
                poly_features.extend([0.0, 0.0, 0.0])
        
        return poly_features
    
    def _create_transformed_features(self, headers, row):
        """Create transformed features"""
        transform_features = []
        
        # Log and reciprocal transforms
        for i in self.component_cols[:5] + self.property_cols[:10]:
            if i < len(row) and isinstance(row[i], (int, float)):
                val = row[i]
                transform_features.extend([
                    math.log1p(abs(val)),  # Log1p
                    1.0 / (abs(val) + 1e-8),  # Reciprocal
                ])
            else:
                transform_features.extend([0.0, 0.0])
        
        return transform_features
    
    def _create_diversity_features(self, headers, row):
        """Create diversity and entropy features"""
        diversity_features = []
        
        # Component diversity
        comp_values = [row[i] if i < len(row) and isinstance(row[i], (int, float)) else 0.0 
                      for i in self.component_cols]
        
        if comp_values:
            total = sum(comp_values) + 1e-8
            normalized = [v / total for v in comp_values]
            
            # Entropy
            entropy = -sum(p * math.log(p + 1e-8) for p in normalized if p > 0)
            
            # Gini coefficient
            gini = 1 - sum(p ** 2 for p in normalized)
            
            # Dominant component ratio
            max_comp = max(normalized) if normalized else 0
            second_max = sorted(normalized, reverse=True)[1] if len(normalized) > 1 else 0
            dominance_ratio = max_comp / (second_max + 1e-8)
            
            diversity_features.extend([entropy, gini, dominance_ratio])
        else:
            diversity_features.extend([0.0, 0.0, 0.0])
        
        return diversity_features
    
    def _generate_feature_names(self, original_headers):
        """Generate names for all engineered features"""
        self.feature_names = original_headers.copy()
        
        # Add component stats
        if self.component_cols:
            self.feature_names.extend([
                'comp_mean', 'comp_std', 'comp_min', 'comp_max', 'comp_median',
                'comp_range', 'comp_cv', 'comp_skew', 'comp_kurt', 'comp_sum'
            ])
        
        # Add property stats
        if self.property_cols:
            self.feature_names.extend([
                'prop_mean', 'prop_std', 'prop_min', 'prop_max', 'prop_median',
                'prop_range', 'prop_cv', 'prop_skew', 'prop_kurt'
            ])
        
        # Add weighted features
        for comp in range(1, 6):
            self.feature_names.extend([
                f'comp{comp}_weighted_avg', f'comp{comp}_weighted_max',
                f'comp{comp}_weighted_min', f'comp{comp}_intensity', f'comp{comp}_weighted_sum'
            ])
        
        # Add interaction features
        n_comps = len(self.component_cols)
        n_interactions = n_comps * (n_comps - 1) // 2
        for i in range(n_interactions):
            self.feature_names.extend([
                f'comp_interact_{i}_prod', f'comp_interact_{i}_diff',
                f'comp_interact_{i}_ratio', f'comp_interact_{i}_avg'
            ])
        
        # Add cross interactions
        for i in range(9):  # 3 comps × 3 props
            self.feature_names.append(f'cross_interact_{i}')
        
        # Add polynomial features
        for i in range(5):
            self.feature_names.extend([f'comp{i}_sq', f'comp{i}_cube', f'comp{i}_sqrt'])
        
        # Add transformed features
        for i in range(15):  # 5 comps + 10 props
            self.feature_names.extend([f'transform_{i}_log1p', f'transform_{i}_recip'])
        
        # Add diversity features
        self.feature_names.extend(['entropy', 'gini', 'dominance_ratio'])


class PropertyWiseRegressor:
    """Property-wise regression with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        
    def ridge_regression_advanced(self, X, y, alpha=1.0):
        """Advanced ridge regression with better conditioning"""
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
            # Add L2 regularization
            XTX[i][i] += alpha
        
        # X^T * y
        XTy = [0.0] * n_features
        for i in range(n_features):
            for k in range(n_samples):
                XTy[i] += X_with_bias[k][i] * y[k]
        
        # Solve using Gaussian elimination with pivoting
        return self._solve_linear_system(XTX, XTy)
    
    def huber_regression(self, X, y, epsilon=1.35, alpha=0.1, max_iter=50):
        """Huber regression for robustness"""
        if not X or not X[0] or not y:
            return [0.0] * (len(X[0]) + 1 if X and X[0] else 1)
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Add bias term
        X_with_bias = [[1.0] + row for row in X]
        n_features += 1
        
        # Initialize weights with ridge regression
        weights = self.ridge_regression_advanced(X, y, alpha)
        
        # Iteratively reweighted least squares
        for iteration in range(max_iter):
            # Compute residuals
            residuals = []
            for i in range(n_samples):
                pred = sum(X_with_bias[i][j] * weights[j] for j in range(n_features))
                residuals.append(y[i] - pred)
            
            # Compute Huber weights
            huber_weights = []
            for r in residuals:
                abs_r = abs(r)
                if abs_r <= epsilon:
                    huber_weights.append(1.0)
                else:
                    huber_weights.append(epsilon / abs_r)
            
            # Weighted normal equations
            WTX = [[0.0] * n_features for _ in range(n_features)]
            WTy = [0.0] * n_features
            
            for i in range(n_features):
                for j in range(n_features):
                    for k in range(n_samples):
                        WTX[i][j] += huber_weights[k] * X_with_bias[k][i] * X_with_bias[k][j]
                # Add regularization
                WTX[i][i] += alpha
                
                for k in range(n_samples):
                    WTy[i] += huber_weights[k] * X_with_bias[k][i] * y[k]
            
            # Solve and update weights
            try:
                new_weights = self._solve_linear_system(WTX, WTy)
                
                # Check convergence
                weight_change = sum(abs(new_weights[i] - weights[i]) for i in range(n_features))
                weights = new_weights
                
                if weight_change < 1e-6:
                    break
            except:
                break
        
        return weights
    
    def ensemble_regression(self, X, y):
        """Ensemble of multiple regression models"""
        # Model 1: Ridge with light regularization
        weights_ridge_light = self.ridge_regression_advanced(X, y, alpha=0.1)
        
        # Model 2: Ridge with medium regularization
        weights_ridge_medium = self.ridge_regression_advanced(X, y, alpha=1.0)
        
        # Model 3: Ridge with strong regularization
        weights_ridge_strong = self.ridge_regression_advanced(X, y, alpha=10.0)
        
        # Model 4: Huber regression
        weights_huber = self.huber_regression(X, y, epsilon=1.35, alpha=1.0)
        
        # Ensemble weights (favor medium regularization and Huber)
        ensemble_weights = [0.15, 0.35, 0.15, 0.35]  # light, medium, strong, huber
        
        # Combine models
        n_features = len(weights_ridge_light)
        final_weights = [0.0] * n_features
        
        all_weights = [weights_ridge_light, weights_ridge_medium, weights_ridge_strong, weights_huber]
        
        for i in range(n_features):
            for j, model_weights in enumerate(all_weights):
                if i < len(model_weights):
                    final_weights[i] += ensemble_weights[j] * model_weights[i]
        
        return final_weights
    
    def cross_validate(self, X, y, k=5):
        """Cross-validation for model evaluation"""
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
            weights = self.ensemble_regression(X_train_fold, y_train_fold)
            
            # Predict and calculate MAE
            mae = 0.0
            for i, x_val in enumerate(X_val):
                pred = sum(weights[j] * ([1.0] + x_val)[j] for j in range(min(len(weights), len(x_val) + 1)))
                mae += abs(pred - y_val[i])
            
            mae /= len(X_val)
            mae_scores.append(mae)
        
        return mae_scores
    
    def train_property_model(self, X, y, property_name):
        """Train model for a specific property"""
        print(f"🎯 Training model for {property_name}")
        
        # Cross-validation
        cv_scores = self.cross_validate(X, y)
        
        # Train final model on all data
        weights = self.ensemble_regression(X, y)
        
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
    
    def _solve_linear_system(self, A, b):
        """Solve linear system Ax = b using Gaussian elimination"""
        n = len(A)
        
        # Create augmented matrix
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        # Forward elimination with partial pivoting
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            
            if max_row != i:
                aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Check for near-zero pivot
            if abs(aug[i][i]) < 1e-12:
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


def robust_scale_data(X_train, X_test):
    """Robust scaling using median and MAD"""
    if not X_train or not X_train[0]:
        return X_train, X_test
    
    n_features = len(X_train[0])
    medians = []
    scales = []
    
    for j in range(n_features):
        feature_values = [row[j] for row in X_train if isinstance(row[j], (int, float))]
        if feature_values:
            sorted_vals = sorted(feature_values)
            median = sorted_vals[len(sorted_vals) // 2]
            
            # MAD (Median Absolute Deviation)
            abs_devs = [abs(x - median) for x in feature_values]
            mad = sorted(abs_devs)[len(abs_devs) // 2]
            
            medians.append(median)
            scales.append(max(mad * 1.4826, 1e-8))  # MAD to std conversion
        else:
            medians.append(0.0)
            scales.append(1.0)
    
    # Scale data
    def scale_data(data):
        scaled_data = []
        for row in data:
            scaled_row = []
            for j, val in enumerate(row):
                if isinstance(val, (int, float)):
                    scaled_val = (val - medians[j]) / scales[j]
                    # Clip extreme outliers
                    scaled_val = max(-5, min(5, scaled_val))
                    scaled_row.append(scaled_val)
                else:
                    scaled_row.append(0.0)
            scaled_data.append(scaled_row)
        return scaled_data
    
    return scale_data(X_train), scale_data(X_test)


def log_experiment(cv_scores, feature_count, notes=""):
    """Log experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_data = {
        'timestamp': timestamp,
        'cv_scores': cv_scores,
        'feature_count': feature_count,
        'notes': notes,
        'avg_mae': sum(scores['mae_mean'] for scores in cv_scores.values()) / len(cv_scores),
        'avg_std': sum(scores['mae_std'] for scores in cv_scores.values()) / len(cv_scores)
    }
    
    # Create logs directory
    os.makedirs('model_logs', exist_ok=True)
    
    # Save to JSON
    log_file = f'model_logs/experiment_{timestamp}.json'
    with open(log_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"📝 Experiment logged to {log_file}")
    print(f"📊 Average MAE: {experiment_data['avg_mae']:.4f}")
    
    return experiment_data


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
    print("🚀 PURE PYTHON ADVANCED SHELL.AI PIPELINE")
    print("=" * 60)
    print("🎯 Property-wise modeling with advanced features")
    print("🔧 Ensemble regression with cross-validation")
    print("📊 Optimized for MAE performance")
    print()
    
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
        
        # Advanced feature engineering
        print("\n🔧 Advanced Feature Engineering")
        feature_engineer = PurePythonAdvancedFeatureEngineer()
        feature_engineer.fit([train_headers[i] for i in feature_indices], X_train_raw)
        
        X_train_engineered = feature_engineer.transform([train_headers[i] for i in feature_indices], X_train_raw)
        X_test_engineered = feature_engineer.transform([train_headers[i] for i in feature_indices], X_test_raw)
        
        # Robust scaling
        print("📊 Robust feature scaling...")
        X_train_scaled, X_test_scaled = robust_scale_data(X_train_engineered, X_test_engineered)
        
        # Property-wise modeling
        print("\n🎯 Property-wise Modeling")
        regressor = PropertyWiseRegressor()
        
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
        
        # Log experiment
        print("\n📝 Logging experiment...")
        experiment_data = log_experiment(
            regressor.cv_scores,
            len(X_train_scaled[0]) if X_train_scaled else 0,
            "Pure Python advanced pipeline with property-wise modeling"
        )
        
        # Create submission
        create_submission(predictions, test_data, test_headers, target_names)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 Features engineered: {len(X_train_scaled[0]) if X_train_scaled else 0}")
        print(f"🎯 Properties modeled: {len(target_names)}")
        print(f"📈 Average CV MAE: {experiment_data['avg_mae']:.4f}")
        print(f"📁 Submission: submission.csv")
        print("🏆 Expected: Significantly improved performance!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()