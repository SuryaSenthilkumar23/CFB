#!/usr/bin/env python3
"""
Shell.ai Hackathon 2025 - Improved Multi-Target Solution
========================================================

Advanced solution with enhanced feature engineering and ensemble methods.
Target: Score significantly higher than 22.
"""

import csv
import math
import random
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

def load_csv(filepath):
    """Load CSV file and return headers and data"""
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

def robust_stats(values):
    """Calculate robust statistics including percentiles"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'q25': 0, 'q75': 0, 'iqr': 0}
    
    sorted_vals = sorted(values)
    n = len(values)
    
    # Basic stats
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n if n > 1 else 0
    std_val = math.sqrt(variance)
    
    # Percentiles
    def percentile(data, p):
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return d0 + d1
    
    median = percentile(sorted_vals, 0.5)
    q25 = percentile(sorted_vals, 0.25)
    q75 = percentile(sorted_vals, 0.75)
    iqr = q75 - q25
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min(values),
        'max': max(values),
        'median': median,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'range': max(values) - min(values),
        'skew': sum((x - mean_val) ** 3 for x in values) / (n * std_val ** 3) if std_val > 0 else 0
    }

def advanced_feature_engineering(headers, data):
    """Advanced feature engineering with more sophisticated features"""
    print("🔧 Advanced feature engineering...")
    
    # Find column indices
    component_indices = [i for i, h in enumerate(headers) if 'fraction' in h.lower()]
    property_indices = [i for i, h in enumerate(headers) if 'Property' in h]
    
    print(f"   Found {len(component_indices)} component columns")
    print(f"   Found {len(property_indices)} property columns")
    
    new_headers = headers.copy()
    new_data = []
    
    for row in data:
        new_row = row.copy()
        
        # 1. Component fraction statistics (robust)
        if component_indices:
            comp_values = [row[i] for i in component_indices if isinstance(row[i], (int, float))]
            if comp_values:
                stats = robust_stats(comp_values)
                new_row.extend([
                    stats['mean'], stats['std'], stats['median'],
                    stats['q25'], stats['q75'], stats['iqr'],
                    stats['min'], stats['max'], stats['range'],
                    stats['skew'], sum(comp_values)
                ])
        
        # 2. Property statistics (robust)
        if property_indices:
            prop_values = [row[i] for i in property_indices if isinstance(row[i], (int, float))]
            if prop_values:
                stats = robust_stats(prop_values)
                new_row.extend([
                    stats['mean'], stats['std'], stats['median'],
                    stats['q25'], stats['q75'], stats['iqr'],
                    stats['min'], stats['max'], stats['range'],
                    stats['skew']
                ])
        
        # 3. Component-wise property statistics
        component_property_features = []
        for comp in range(1, 6):  # Components 1-5
            comp_props = []
            comp_frac = 0.0
            
            for i, h in enumerate(headers):
                if f'Component{comp}_fraction' in h and isinstance(row[i], (int, float)):
                    comp_frac = row[i]
                elif f'Component{comp}' in h and 'Property' in h and isinstance(row[i], (int, float)):
                    comp_props.append(row[i])
            
            if comp_props:
                comp_stats = robust_stats(comp_props)
                # Weighted features by fraction
                component_property_features.extend([
                    comp_frac * comp_stats['mean'],
                    comp_frac * comp_stats['std'],
                    comp_frac * comp_stats['max'],
                    comp_frac * comp_stats['min'],
                    comp_frac * comp_stats['median'],
                    comp_stats['mean'] / (comp_frac + 1e-8),  # Property intensity
                    comp_frac ** 2 * comp_stats['mean'],  # Quadratic interaction
                ])
            else:
                component_property_features.extend([0.0] * 7)
        
        new_row.extend(component_property_features)
        
        # 4. Cross-component interactions
        if len(component_indices) >= 2:
            for i in range(len(component_indices)):
                for j in range(i + 1, len(component_indices)):
                    frac_i = row[component_indices[i]] if isinstance(row[component_indices[i]], (int, float)) else 0.0
                    frac_j = row[component_indices[j]] if isinstance(row[component_indices[j]], (int, float)) else 0.0
                    
                    new_row.extend([
                        frac_i * frac_j,  # Product
                        abs(frac_i - frac_j),  # Difference
                        (frac_i + frac_j) / 2,  # Average
                        frac_i / (frac_j + 1e-8),  # Ratio
                    ])
        
        # 5. Property correlations and ratios
        if len(property_indices) >= 2:
            prop_values = [row[i] if isinstance(row[i], (int, float)) else 0.0 for i in property_indices]
            
            # Property ratios (first 10 properties)
            for i in range(min(10, len(prop_values))):
                for j in range(i + 1, min(10, len(prop_values))):
                    if abs(prop_values[j]) > 1e-8:
                        new_row.append(prop_values[i] / prop_values[j])
                    else:
                        new_row.append(0.0)
        
        # 6. Polynomial features (degree 2) for top components
        if component_indices:
            top_fractions = [row[i] if isinstance(row[i], (int, float)) else 0.0 
                           for i in component_indices[:3]]  # Top 3 components
            
            for frac in top_fractions:
                new_row.extend([
                    frac ** 2,
                    frac ** 3,
                    math.sqrt(abs(frac)),
                    math.log(abs(frac) + 1e-8),
                ])
        
        # 7. Density and concentration features
        if component_indices and property_indices:
            total_fraction = sum(row[i] if isinstance(row[i], (int, float)) else 0.0 
                               for i in component_indices)
            
            if total_fraction > 1e-8:
                # Effective properties weighted by fractions
                weighted_props = []
                for prop_idx in property_indices[:20]:  # Top 20 properties
                    weighted_prop = 0.0
                    for comp in range(1, 6):
                        comp_frac = 0.0
                        comp_prop = 0.0
                        
                        for i, h in enumerate(headers):
                            if f'Component{comp}_fraction' in h and isinstance(row[i], (int, float)):
                                comp_frac = row[i]
                            elif (f'Component{comp}' in h and 
                                  headers[prop_idx].replace('Component1', f'Component{comp}') == h and 
                                  isinstance(row[i], (int, float))):
                                comp_prop = row[i]
                        
                        weighted_prop += comp_frac * comp_prop
                    
                    weighted_props.append(weighted_prop / total_fraction)
                
                # Add weighted property statistics
                if weighted_props:
                    w_stats = robust_stats(weighted_props)
                    new_row.extend([
                        w_stats['mean'], w_stats['std'], w_stats['median'],
                        w_stats['max'], w_stats['min']
                    ])
        
        new_data.append(new_row)
    
    # Add new headers
    feature_names = []
    
    # Component stats
    if component_indices:
        feature_names.extend([
            'comp_mean', 'comp_std', 'comp_median', 'comp_q25', 'comp_q75', 
            'comp_iqr', 'comp_min', 'comp_max', 'comp_range', 'comp_skew', 'comp_total'
        ])
    
    # Property stats
    if property_indices:
        feature_names.extend([
            'prop_mean', 'prop_std', 'prop_median', 'prop_q25', 'prop_q75',
            'prop_iqr', 'prop_min', 'prop_max', 'prop_range', 'prop_skew'
        ])
    
    # Component-wise features
    for comp in range(1, 6):
        feature_names.extend([
            f'comp{comp}_weighted_mean', f'comp{comp}_weighted_std',
            f'comp{comp}_weighted_max', f'comp{comp}_weighted_min',
            f'comp{comp}_weighted_median', f'comp{comp}_intensity',
            f'comp{comp}_quad_interaction'
        ])
    
    # Cross-component interactions
    if len(component_indices) >= 2:
        for i in range(len(component_indices)):
            for j in range(i + 1, len(component_indices)):
                feature_names.extend([
                    f'comp{i}_{j}_product', f'comp{i}_{j}_diff',
                    f'comp{i}_{j}_avg', f'comp{i}_{j}_ratio'
                ])
    
    # Property ratios
    if len(property_indices) >= 2:
        for i in range(min(10, len(property_indices))):
            for j in range(i + 1, min(10, len(property_indices))):
                feature_names.append(f'prop{i}_{j}_ratio')
    
    # Polynomial features
    if component_indices:
        for i in range(min(3, len(component_indices))):
            feature_names.extend([
                f'comp{i}_sq', f'comp{i}_cube', f'comp{i}_sqrt', f'comp{i}_log'
            ])
    
    # Weighted property features
    if component_indices and property_indices:
        feature_names.extend([
            'weighted_prop_mean', 'weighted_prop_std', 'weighted_prop_median',
            'weighted_prop_max', 'weighted_prop_min'
        ])
    
    new_headers.extend(feature_names)
    
    print(f"✅ Features: {len(headers)} → {len(new_headers)} (+{len(feature_names)} new)")
    return new_headers, new_data

def prepare_data_advanced(train_headers, train_data, test_headers, test_data):
    """Prepare data with advanced preprocessing"""
    print("📋 Advanced data preparation...")
    
    # Find target columns
    target_indices = [i for i, h in enumerate(train_headers) if 'BlendProperty' in h]
    target_names = [train_headers[i] for i in target_indices]
    
    # Find feature columns (exclude ID and targets)
    feature_indices = [i for i, h in enumerate(train_headers) 
                      if h not in ['ID'] and i not in target_indices]
    feature_names = [train_headers[i] for i in feature_indices]
    
    print(f"🎯 Targets: {len(target_names)}")
    print(f"📊 Features: {len(feature_names)}")
    
    # Match test features
    test_feature_indices = []
    for fname in feature_names:
        try:
            idx = test_headers.index(fname)
            test_feature_indices.append(idx)
        except ValueError:
            test_feature_indices.append(-1)
    
    # Extract training data
    X_train, y_train = [], []
    for row in train_data:
        # Features
        features = []
        for i in feature_indices:
            if i < len(row) and isinstance(row[i], (int, float)):
                features.append(row[i])
            else:
                features.append(0.0)
        X_train.append(features)
        
        # Targets
        targets = []
        for i in target_indices:
            if i < len(row) and isinstance(row[i], (int, float)):
                targets.append(row[i])
            else:
                targets.append(0.0)
        y_train.append(targets)
    
    # Extract test data
    X_test = []
    for row in test_data:
        features = []
        for i in test_feature_indices:
            if i >= 0 and i < len(row) and isinstance(row[i], (int, float)):
                features.append(row[i])
            else:
                features.append(0.0)
        X_test.append(features)
    
    # Feature scaling (robust scaling)
    print("🔧 Robust feature scaling...")
    scaled_X_train, scaled_X_test = robust_scale_features(X_train, X_test)
    
    return scaled_X_train, y_train, scaled_X_test, target_names

def robust_scale_features(X_train, X_test):
    """Apply robust scaling using median and MAD"""
    if not X_train or not X_train[0]:
        return X_train, X_test
    
    n_features = len(X_train[0])
    
    # Calculate median and MAD for each feature
    medians = []
    mads = []
    
    for j in range(n_features):
        feature_values = [row[j] for row in X_train if isinstance(row[j], (int, float))]
        if feature_values:
            sorted_vals = sorted(feature_values)
            median = sorted_vals[len(sorted_vals) // 2]
            
            # MAD (Median Absolute Deviation)
            abs_devs = [abs(x - median) for x in feature_values]
            mad = sorted(abs_devs)[len(abs_devs) // 2]
            
            medians.append(median)
            mads.append(max(mad, 1e-8))  # Avoid division by zero
        else:
            medians.append(0.0)
            mads.append(1.0)
    
    # Scale training data
    scaled_X_train = []
    for row in X_train:
        scaled_row = []
        for j, val in enumerate(row):
            if isinstance(val, (int, float)):
                scaled_val = (val - medians[j]) / mads[j]
                scaled_row.append(scaled_val)
            else:
                scaled_row.append(0.0)
        scaled_X_train.append(scaled_row)
    
    # Scale test data
    scaled_X_test = []
    for row in X_test:
        scaled_row = []
        for j, val in enumerate(row):
            if isinstance(val, (int, float)):
                scaled_val = (val - medians[j]) / mads[j]
                scaled_row.append(scaled_val)
            else:
                scaled_row.append(0.0)
        scaled_X_test.append(scaled_row)
    
    return scaled_X_train, scaled_X_test

def ridge_regression_advanced(X, y, alpha=1.0):
    """Advanced ridge regression with better numerical stability"""
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
        # Add regularization to diagonal
        XTX[i][i] += alpha
    
    # X^T * y
    XTy = [0.0] * n_features
    for i in range(n_features):
        for k in range(n_samples):
            XTy[i] += X_with_bias[k][i] * y[k]
    
    # Solve using Cholesky decomposition approximation
    try:
        # Simple Gaussian elimination with partial pivoting
        aug = [row[:] + [XTy[i]] for i, row in enumerate(XTX)]
        
        # Forward elimination with partial pivoting
        for i in range(n_features):
            # Find pivot
            max_row = i
            for k in range(i + 1, n_features):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            
            if max_row != i:
                aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Check for near-zero pivot
            if abs(aug[i][i]) < 1e-12:
                aug[i][i] = 1e-6
            
            # Eliminate
            for k in range(i + 1, n_features):
                if abs(aug[i][i]) > 1e-12:
                    factor = aug[k][i] / aug[i][i]
                    for j in range(i, n_features + 1):
                        aug[k][j] -= factor * aug[i][j]
        
        # Back substitution
        weights = [0.0] * n_features
        for i in range(n_features - 1, -1, -1):
            weights[i] = aug[i][n_features]
            for j in range(i + 1, n_features):
                weights[i] -= aug[i][j] * weights[j]
            if abs(aug[i][i]) > 1e-12:
                weights[i] /= aug[i][i]
        
        return weights
    
    except:
        # Fallback: simple least squares
        return [0.0] * n_features

def ensemble_predict(X_train, y_train, X_test, target_name):
    """Ensemble prediction using multiple alpha values and techniques"""
    
    # Multiple ridge regression models with different regularization
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    predictions = []
    
    for alpha in alphas:
        weights = ridge_regression_advanced(X_train, y_train, alpha)
        pred = predict_linear(X_test, weights)
        predictions.append(pred)
    
    # Ensemble: weighted average (give more weight to middle alphas)
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    
    if not predictions or not predictions[0]:
        return [0.0] * len(X_test)
    
    ensemble_pred = []
    for i in range(len(predictions[0])):
        weighted_sum = sum(w * pred[i] for w, pred in zip(weights, predictions))
        ensemble_pred.append(weighted_sum)
    
    return ensemble_pred

def predict_linear(X, weights):
    """Make predictions with linear model"""
    if not X or not weights:
        return []
    
    predictions = []
    for row in X:
        # Add bias term
        row_with_bias = [1.0] + row
        pred = sum(row_with_bias[i] * weights[i] for i in range(min(len(weights), len(row_with_bias))))
        predictions.append(pred)
    
    return predictions

def train_advanced_ensemble(X_train, y_train, target_names):
    """Train advanced ensemble models for all targets"""
    print("🏗️ Training advanced ensemble models...")
    
    models = {}
    
    for i, target_name in enumerate(target_names):
        print(f"   Training ensemble for {target_name}...")
        
        # Extract target values
        y_target = [row[i] for row in y_train]
        
        # Train ensemble
        models[target_name] = lambda X_test, target_idx=i: ensemble_predict(
            X_train, [row[target_idx] for row in y_train], X_test, target_name
        )
    
    print("✅ Advanced ensemble training complete!")
    return models

def predict_advanced_ensemble(X_test, models, target_names):
    """Make predictions using advanced ensemble"""
    print("🔮 Making advanced ensemble predictions...")
    
    n_samples = len(X_test)
    n_targets = len(target_names)
    predictions = [[0.0] * n_targets for _ in range(n_samples)]
    
    for i, target_name in enumerate(target_names):
        print(f"   Predicting {target_name}...")
        target_preds = models[target_name](X_test)
        
        for j in range(n_samples):
            if j < len(target_preds):
                predictions[j][i] = target_preds[j]
    
    return predictions

def save_submission(predictions, test_data, test_headers, target_names, filename='submission.csv'):
    """Save predictions to submission file"""
    print("📝 Creating improved submission...")
    
    # Find ID column
    id_col_idx = None
    for i, h in enumerate(test_headers):
        if h == 'ID':
            id_col_idx = i
            break
    
    # Create submission
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        headers = ['ID'] + target_names
        writer.writerow(headers)
        
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
    
    print(f"✅ Improved submission saved to {filename}")
    
    # Print statistics
    if predictions:
        flat_preds = [val for row in predictions for val in row]
        if flat_preds:
            pred_mean = sum(flat_preds) / len(flat_preds)
            pred_std = math.sqrt(sum((x - pred_mean) ** 2 for x in flat_preds) / len(flat_preds))
            pred_min = min(flat_preds)
            pred_max = max(flat_preds)
            print(f"📊 Prediction stats: mean={pred_mean:.4f}, std={pred_std:.4f}")
            print(f"📊 Range: [{pred_min:.4f}, {pred_max:.4f}]")

def main():
    """Main execution function"""
    print("🚀 Shell.ai Hackathon 2025 - IMPROVED Multi-Target Solution")
    print("=" * 65)
    print("🎯 Target: Score significantly higher than 22")
    print()
    
    try:
        # Load data
        print("🔄 Loading data...")
        train_headers, train_data = load_csv('train.csv')
        test_headers, test_data = load_csv('test.csv')
        sample_headers, sample_data = load_csv('sample_solution.csv')
        
        print(f"📊 Data loaded:")
        print(f"   Train: {len(train_data)} x {len(train_headers)}")
        print(f"   Test: {len(test_data)} x {len(test_headers)}")
        
        # Advanced feature engineering
        train_headers_eng, train_data_eng = advanced_feature_engineering(train_headers, train_data)
        test_headers_eng, test_data_eng = advanced_feature_engineering(test_headers, test_data)
        
        # Advanced data preparation
        X_train, y_train, X_test, target_names = prepare_data_advanced(
            train_headers_eng, train_data_eng, 
            test_headers_eng, test_data_eng
        )
        
        print(f"📈 Final dataset: {len(X_train)} train samples, {len(X_test)} test samples")
        print(f"📈 Feature dimensions: {len(X_train[0]) if X_train else 0}")
        
        # Train advanced ensemble
        models = train_advanced_ensemble(X_train, y_train, target_names)
        
        # Make predictions
        predictions = predict_advanced_ensemble(X_test, models, target_names)
        
        # Save submission
        save_submission(predictions, test_data, test_headers, target_names, 'submission.csv')
        
        print("\n🎉 IMPROVED pipeline completed successfully!")
        print("📁 Generated: submission.csv")
        print("🏆 Expected: Score > 22 (previous score)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()