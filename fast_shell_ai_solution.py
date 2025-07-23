#!/usr/bin/env python3
"""
Shell.ai Hackathon 2025 - Fast Multi-Target Solution
====================================================

Quick and effective solution for fuel blend property prediction.
Optimized for speed while maintaining good accuracy.
"""

import csv
import math
import random

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

def simple_stats(values):
    """Calculate basic statistics"""
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

def engineer_features(headers, data):
    """Fast feature engineering"""
    print("🔧 Engineering features...")
    
    # Find column indices
    component_indices = [i for i, h in enumerate(headers) if 'fraction' in h.lower()]
    property_indices = [i for i, h in enumerate(headers) if 'Property' in h]
    
    print(f"   Found {len(component_indices)} component columns")
    print(f"   Found {len(property_indices)} property columns")
    
    new_headers = headers.copy()
    new_data = []
    
    for row in data:
        new_row = row.copy()
        
        # Component statistics
        if component_indices:
            comp_values = [row[i] for i in component_indices if isinstance(row[i], (int, float))]
            if comp_values:
                stats = simple_stats(comp_values)
                new_row.extend([
                    sum(comp_values),  # total
                    stats['max'],      # max
                    stats['min'],      # min
                    stats['std'],      # std
                ])
        
        # Property statistics
        if property_indices:
            prop_values = [row[i] for i in property_indices if isinstance(row[i], (int, float))]
            if prop_values:
                stats = simple_stats(prop_values)
                new_row.extend([
                    stats['mean'],     # mean
                    stats['std'],      # std
                    stats['max'],      # max
                    stats['min'],      # min
                ])
        
        # Simple interactions (component * avg property for each component)
        for comp in range(1, 6):  # Components 1-5
            comp_frac = None
            comp_props = []
            
            for i, h in enumerate(headers):
                if f'Component{comp}_fraction' in h and isinstance(row[i], (int, float)):
                    comp_frac = row[i]
                elif f'Component{comp}' in h and 'Property' in h and isinstance(row[i], (int, float)):
                    comp_props.append(row[i])
            
            if comp_frac is not None and comp_props:
                avg_prop = sum(comp_props) / len(comp_props)
                interaction = comp_frac * avg_prop
                new_row.append(interaction)
        
        new_data.append(new_row)
    
    # Add new headers
    if component_indices:
        new_headers.extend(['total_comp', 'max_comp', 'min_comp', 'std_comp'])
    if property_indices:
        new_headers.extend(['mean_prop', 'std_prop', 'max_prop', 'min_prop'])
    for comp in range(1, 6):
        new_headers.append(f'interaction_comp{comp}')
    
    print(f"✅ Features: {len(headers)} → {len(new_headers)}")
    return new_headers, new_data

def prepare_data(train_headers, train_data, test_headers, test_data):
    """Prepare features and targets"""
    print("📋 Preparing data...")
    
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
    
    return X_train, y_train, X_test, target_names

def simple_linear_regression(X, y):
    """Fast linear regression using normal equations"""
    if not X or not X[0] or not y:
        return [0.0] * (len(X[0]) if X and X[0] else 1)
    
    n_samples = len(X)
    n_features = len(X[0])
    
    # Add bias term
    X_with_bias = [[1.0] + row for row in X]
    n_features += 1
    
    # X^T * X
    XTX = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            for k in range(n_samples):
                XTX[i][j] += X_with_bias[k][i] * X_with_bias[k][j]
    
    # X^T * y
    XTy = [0.0] * n_features
    for i in range(n_features):
        for k in range(n_samples):
            XTy[i] += X_with_bias[k][i] * y[k]
    
    # Add regularization to diagonal
    alpha = 1.0
    for i in range(n_features):
        XTX[i][i] += alpha
    
    # Solve using simple Gaussian elimination
    try:
        # Create augmented matrix
        aug = [row[:] + [XTy[i]] for i, row in enumerate(XTX)]
        
        # Forward elimination
        for i in range(n_features):
            # Find pivot
            max_row = i
            for k in range(i + 1, n_features):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Eliminate
            for k in range(i + 1, n_features):
                if aug[i][i] != 0:
                    factor = aug[k][i] / aug[i][i]
                    for j in range(i, n_features + 1):
                        aug[k][j] -= factor * aug[i][j]
        
        # Back substitution
        weights = [0.0] * n_features
        for i in range(n_features - 1, -1, -1):
            weights[i] = aug[i][n_features]
            for j in range(i + 1, n_features):
                weights[i] -= aug[i][j] * weights[j]
            if aug[i][i] != 0:
                weights[i] /= aug[i][i]
        
        return weights
    
    except:
        # Fallback to zero weights
        return [0.0] * n_features

def predict_linear(X, weights):
    """Make predictions with linear model"""
    if not X or not weights:
        return []
    
    predictions = []
    for row in X:
        # Add bias term
        row_with_bias = [1.0] + row
        pred = sum(row_with_bias[i] * weights[i] for i in range(len(weights)))
        predictions.append(pred)
    
    return predictions

def train_multi_target_model(X_train, y_train, target_names):
    """Train models for all targets"""
    print("🏗️ Training multi-target models...")
    
    models = {}
    n_targets = len(target_names)
    
    for i, target_name in enumerate(target_names):
        print(f"   Training {target_name}...")
        
        # Extract target values
        y_target = [row[i] for row in y_train]
        
        # Train simple linear regression
        weights = simple_linear_regression(X_train, y_target)
        models[target_name] = weights
    
    print("✅ Training complete!")
    return models

def predict_multi_target(X_test, models, target_names):
    """Make predictions for all targets"""
    print("🔮 Making predictions...")
    
    n_samples = len(X_test)
    n_targets = len(target_names)
    predictions = [[0.0] * n_targets for _ in range(n_samples)]
    
    for i, target_name in enumerate(target_names):
        weights = models[target_name]
        target_preds = predict_linear(X_test, weights)
        
        for j in range(n_samples):
            if j < len(target_preds):
                predictions[j][i] = target_preds[j]
    
    return predictions

def save_submission(predictions, test_data, test_headers, target_names, filename='submission.csv'):
    """Save predictions to submission file"""
    print("📝 Creating submission...")
    
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
    
    print(f"✅ Submission saved to {filename}")
    
    # Print statistics
    if predictions:
        flat_preds = [val for row in predictions for val in row]
        if flat_preds:
            pred_mean = sum(flat_preds) / len(flat_preds)
            pred_min = min(flat_preds)
            pred_max = max(flat_preds)
            print(f"📊 Predictions: mean={pred_mean:.3f}, range=[{pred_min:.3f}, {pred_max:.3f}]")

def main():
    """Main execution function"""
    print("🚀 Shell.ai Hackathon 2025 - Fast Multi-Target Solution")
    print("=" * 60)
    
    try:
        # Load data
        print("🔄 Loading data...")
        train_headers, train_data = load_csv('train.csv')
        test_headers, test_data = load_csv('test.csv')
        sample_headers, sample_data = load_csv('sample_solution.csv')
        
        print(f"📊 Data loaded:")
        print(f"   Train: {len(train_data)} x {len(train_headers)}")
        print(f"   Test: {len(test_data)} x {len(test_headers)}")
        
        # Feature engineering
        train_headers_eng, train_data_eng = engineer_features(train_headers, train_data)
        test_headers_eng, test_data_eng = engineer_features(test_headers, test_data)
        
        # Prepare data
        X_train, y_train, X_test, target_names = prepare_data(
            train_headers_eng, train_data_eng, 
            test_headers_eng, test_data_eng
        )
        
        # Train models
        models = train_multi_target_model(X_train, y_train, target_names)
        
        # Make predictions
        predictions = predict_multi_target(X_test, models, target_names)
        
        # Save submission
        save_submission(predictions, test_data, test_headers, target_names, 'submission.csv')
        
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Generated: submission.csv")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()