#!/usr/bin/env python3
"""
Shell.ai Hackathon 2025 - FOCUSED Improvement Solution
======================================================

Building on what worked (score 22) with carefully selected improvements:
- Start with the successful base approach
- Add only the most proven effective features
- Avoid overfitting with simpler, more robust techniques
- Focus on prediction quality over feature quantity
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
    """Calculate essential statistics"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n if n > 1 else 0
    std_val = math.sqrt(variance)
    
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2]
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min(values),
        'max': max(values),
        'median': median,
        'range': max(values) - min(values)
    }

def focused_feature_engineering(headers, data):
    """Focused feature engineering - build on what worked, add proven improvements"""
    print("🎯 Focused Feature Engineering (Building on Score 22)...")
    
    # Find column indices
    component_indices = [i for i, h in enumerate(headers) if 'fraction' in h.lower()]
    property_indices = [i for i, h in enumerate(headers) if 'Property' in h]
    
    print(f"   Components: {len(component_indices)}, Properties: {len(property_indices)}")
    
    new_headers = headers.copy()
    new_data = []
    
    # Pre-compute component-property mapping for efficiency
    comp_frac_map = {}
    comp_prop_map = {}
    
    for i, h in enumerate(headers):
        for comp in range(1, 6):
            if f'Component{comp}_fraction' in h:
                comp_frac_map[comp] = i
            elif f'Component{comp}' in h and 'Property' in h:
                if comp not in comp_prop_map:
                    comp_prop_map[comp] = []
                comp_prop_map[comp].append(i)
    
    print("🔧 Processing focused features...")
    
    for row_idx, row in enumerate(data):
        if row_idx % 1000 == 0:
            print(f"   Processing row {row_idx}/{len(data)}")
        
        new_row = row.copy()
        
        # 1. KEEP: Basic component and property statistics (these worked!)
        if component_indices:
            comp_values = [row[i] for i in component_indices if isinstance(row[i], (int, float))]
            if comp_values:
                stats = simple_stats(comp_values)
                new_row.extend([
                    stats['mean'], stats['std'], stats['median'],
                    stats['min'], stats['max'], stats['range'],
                    sum(comp_values)  # Total fraction
                ])
        
        if property_indices:
            prop_values = [row[i] for i in property_indices if isinstance(row[i], (int, float))]
            if prop_values:
                stats = simple_stats(prop_values)
                new_row.extend([
                    stats['mean'], stats['std'], stats['median'],
                    stats['min'], stats['max'], stats['range']
                ])
        
        # 2. IMPROVED: Component-wise weighted features (but simpler)
        for comp in range(1, 6):
            comp_frac = 0.0
            comp_props = []
            
            # Get component fraction
            frac_idx = comp_frac_map.get(comp)
            if frac_idx is not None and frac_idx < len(row):
                comp_frac = row[frac_idx] if isinstance(row[frac_idx], (int, float)) else 0.0
            
            # Get component properties (limit to first 10 for stability)
            comp_prop_indices = comp_prop_map.get(comp, [])[:10]
            for prop_idx in comp_prop_indices:
                if prop_idx < len(row) and isinstance(row[prop_idx], (int, float)):
                    comp_props.append(row[prop_idx])
            
            if comp_props:
                # Simple weighted features
                avg_prop = sum(comp_props) / len(comp_props)
                max_prop = max(comp_props)
                min_prop = min(comp_props)
                
                new_row.extend([
                    comp_frac * avg_prop,     # Weighted average
                    comp_frac * max_prop,     # Weighted max
                    comp_frac * min_prop,     # Weighted min
                    avg_prop / (comp_frac + 1e-8)  # Property intensity
                ])
            else:
                new_row.extend([0.0] * 4)
        
        # 3. NEW: Simple pairwise component interactions (only most important)
        pairwise_features = []
        for i in range(len(component_indices)):
            for j in range(i + 1, len(component_indices)):
                frac_i = row[component_indices[i]] if isinstance(row[component_indices[i]], (int, float)) else 0.0
                frac_j = row[component_indices[j]] if isinstance(row[component_indices[j]], (int, float)) else 0.0
                
                pairwise_features.extend([
                    frac_i * frac_j,                    # Product
                    abs(frac_i - frac_j),              # Difference
                    (frac_i + frac_j) / 2,             # Average
                ])
        
        new_row.extend(pairwise_features)
        
        # 4. NEW: Essential nonlinear transforms (only for top 5 components)
        nonlinear_features = []
        for i in component_indices[:5]:  # Top 5 components only
            if i < len(row) and isinstance(row[i], (int, float)):
                val = row[i]
                nonlinear_features.extend([
                    val ** 2,                    # Square
                    math.sqrt(abs(val)),         # Square root
                    math.log1p(abs(val))        # Log1p
                ])
        
        new_row.extend(nonlinear_features)
        
        # 5. NEW: Weighted property aggregates (simplified)
        total_fraction = sum(row[i] if isinstance(row[i], (int, float)) else 0.0 
                           for i in component_indices)
        
        if total_fraction > 1e-8:
            # Calculate weighted average of first 20 properties
            weighted_props = []
            for prop_idx in property_indices[:20]:  # Limit to 20 for stability
                weighted_val = 0.0
                prop_count = 0
                
                for comp in range(1, 6):
                    frac_idx = comp_frac_map.get(comp)
                    if frac_idx is not None and frac_idx < len(row):
                        comp_frac = row[frac_idx] if isinstance(row[frac_idx], (int, float)) else 0.0
                        
                        # Find matching property for this component
                        comp_props = comp_prop_map.get(comp, [])
                        for comp_prop_idx in comp_props:
                            if comp_prop_idx < len(row):
                                prop_val = row[comp_prop_idx] if isinstance(row[comp_prop_idx], (int, float)) else 0.0
                                weighted_val += (comp_frac / total_fraction) * prop_val
                                prop_count += 1
                                break
                
                if prop_count > 0:
                    weighted_props.append(weighted_val)
            
            # Weighted property statistics
            if weighted_props:
                w_stats = simple_stats(weighted_props)
                new_row.extend([
                    w_stats['mean'], w_stats['std'], 
                    w_stats['min'], w_stats['max']
                ])
            else:
                new_row.extend([0.0] * 4)
        else:
            new_row.extend([0.0] * 4)
        
        new_data.append(new_row)
    
    # Generate feature names
    feature_names = []
    
    # Basic component stats
    if component_indices:
        feature_names.extend([
            'comp_mean', 'comp_std', 'comp_median', 
            'comp_min', 'comp_max', 'comp_range', 'comp_total'
        ])
    
    # Basic property stats
    if property_indices:
        feature_names.extend([
            'prop_mean', 'prop_std', 'prop_median',
            'prop_min', 'prop_max', 'prop_range'
        ])
    
    # Component-wise features
    for comp in range(1, 6):
        feature_names.extend([
            f'comp{comp}_weighted_avg', f'comp{comp}_weighted_max',
            f'comp{comp}_weighted_min', f'comp{comp}_intensity'
        ])
    
    # Pairwise interactions
    n_pairs = len(component_indices) * (len(component_indices) - 1) // 2
    for i in range(n_pairs):
        feature_names.extend([f'pair{i}_product', f'pair{i}_diff', f'pair{i}_avg'])
    
    # Nonlinear transforms
    for i in range(5):
        feature_names.extend([f'comp{i}_sq', f'comp{i}_sqrt', f'comp{i}_log1p'])
    
    # Weighted aggregates
    feature_names.extend(['weighted_prop_mean', 'weighted_prop_std', 
                         'weighted_prop_min', 'weighted_prop_max'])
    
    new_headers.extend(feature_names)
    
    print(f"✅ Features: {len(headers)} → {len(new_headers)} (+{len(feature_names)} focused)")
    return new_headers, new_data

def improved_ensemble_models(X_train, y_train, X_test):
    """Improved ensemble with multiple regularization levels"""
    
    # Model 1: Light regularization
    weights_1 = ridge_regression_stable(X_train, y_train, alpha=0.1)
    pred_1 = predict_linear(X_test, weights_1)
    
    # Model 2: Medium regularization  
    weights_2 = ridge_regression_stable(X_train, y_train, alpha=1.0)
    pred_2 = predict_linear(X_test, weights_2)
    
    # Model 3: Strong regularization
    weights_3 = ridge_regression_stable(X_train, y_train, alpha=10.0)
    pred_3 = predict_linear(X_test, weights_3)
    
    # Weighted ensemble (favor medium regularization)
    if not pred_1 or not pred_2 or not pred_3:
        return [0.0] * len(X_test)
    
    ensemble_pred = []
    for i in range(len(pred_1)):
        # Weighted average: 25% light, 50% medium, 25% strong
        weighted_pred = 0.25 * pred_1[i] + 0.50 * pred_2[i] + 0.25 * pred_3[i]
        ensemble_pred.append(weighted_pred)
    
    return ensemble_pred

def ridge_regression_stable(X, y, alpha=1.0):
    """Stable ridge regression with better numerical conditioning"""
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
    
    # Solve using Gaussian elimination with partial pivoting
    try:
        aug = [row[:] + [XTy[i]] for i, row in enumerate(XTX)]
        
        for i in range(n_features):
            # Find pivot
            max_row = i
            for k in range(i + 1, n_features):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            
            if max_row != i:
                aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Check for near-zero pivot
            if abs(aug[i][i]) < 1e-10:
                aug[i][i] = 1e-6  # Regularize
            
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
        # Fallback to simple solution
        return [0.0] * n_features

def predict_linear(X, weights):
    """Make predictions with linear model"""
    if not X or not weights:
        return []
    
    predictions = []
    for row in X:
        row_with_bias = [1.0] + row
        pred = sum(row_with_bias[i] * weights[i] for i in range(min(len(weights), len(row_with_bias))))
        predictions.append(pred)
    
    return predictions

def robust_scale_features_simple(X_train, X_test):
    """Simple robust scaling"""
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
            
            # Use IQR for scaling
            q25 = sorted_vals[len(sorted_vals) // 4]
            q75 = sorted_vals[3 * len(sorted_vals) // 4]
            scale = max(q75 - q25, 1e-8)
            
            medians.append(median)
            scales.append(scale)
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
                    # Gentle clipping
                    scaled_val = max(-3, min(3, scaled_val))
                    scaled_row.append(scaled_val)
                else:
                    scaled_row.append(0.0)
            scaled_data.append(scaled_row)
        return scaled_data
    
    return scale_data(X_train), scale_data(X_test)

def prepare_data_focused(train_headers, train_data, test_headers, test_data):
    """Focused data preparation"""
    print("📋 Focused data preparation...")
    
    # Find target and feature columns
    target_indices = [i for i, h in enumerate(train_headers) if 'BlendProperty' in h]
    target_names = [train_headers[i] for i in target_indices]
    
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
    
    # Extract data
    X_train, y_train = [], []
    for row in train_data:
        features = [row[i] if i < len(row) and isinstance(row[i], (int, float)) else 0.0 
                   for i in feature_indices]
        targets = [row[i] if i < len(row) and isinstance(row[i], (int, float)) else 0.0 
                  for i in target_indices]
        X_train.append(features)
        y_train.append(targets)
    
    X_test = []
    for row in test_data:
        features = []
        for i in test_feature_indices:
            if i >= 0 and i < len(row) and isinstance(row[i], (int, float)):
                features.append(row[i])
            else:
                features.append(0.0)
        X_test.append(features)
    
    # Simple robust scaling
    print("🔧 Simple robust scaling...")
    scaled_X_train, scaled_X_test = robust_scale_features_simple(X_train, X_test)
    
    return scaled_X_train, y_train, scaled_X_test, target_names

def predict_with_conservative_bounds(X_train, y_train, X_test, target_names):
    """Make predictions with conservative bounds"""
    print("🔮 Making predictions with conservative bounds...")
    
    # Calculate conservative target bounds
    target_bounds = {}
    for i, target_name in enumerate(target_names):
        target_values = [row[i] for row in y_train if isinstance(row[i], (int, float))]
        if target_values:
            mean_val = sum(target_values) / len(target_values)
            std_val = math.sqrt(sum((x - mean_val) ** 2 for x in target_values) / len(target_values))
            
            target_bounds[target_name] = {
                'min': min(target_values),
                'max': max(target_values),
                'mean': mean_val,
                'std': std_val
            }
        else:
            target_bounds[target_name] = {'min': -2, 'max': 2, 'mean': 0, 'std': 1}
    
    print("📊 Target bounds:")
    for name, bounds in target_bounds.items():
        print(f"   {name}: [{bounds['min']:.3f}, {bounds['max']:.3f}]")
    
    # Predict each target
    n_samples = len(X_test)
    n_targets = len(target_names)
    predictions = [[0.0] * n_targets for _ in range(n_samples)]
    
    for i, target_name in enumerate(target_names):
        print(f"   Predicting {target_name}...")
        
        # Extract target values
        y_target = [row[i] for row in y_train]
        
        # Get ensemble predictions
        target_preds = improved_ensemble_models(X_train, y_target, X_test)
        
        # Apply conservative bounds
        bounds = target_bounds[target_name]
        
        for j in range(n_samples):
            if j < len(target_preds):
                pred = target_preds[j]
                
                # Check for invalid values
                if not isinstance(pred, (int, float)) or math.isnan(pred) or math.isinf(pred):
                    pred = bounds['mean']
                
                # Conservative clipping (smaller margin)
                margin = (bounds['max'] - bounds['min']) * 0.05  # Reduced margin
                pred = max(bounds['min'] - margin, min(bounds['max'] + margin, pred))
                
                predictions[j][i] = pred
    
    return predictions

def save_focused_submission(predictions, test_data, test_headers, target_names, filename='submission.csv'):
    """Save predictions with validation"""
    print("📝 Creating focused submission...")
    
    # Find ID column
    id_col_idx = test_headers.index('ID') if 'ID' in test_headers else None
    
    # Create submission
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
            
            # Add predictions (validate)
            for val in pred_row:
                if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
                    submission_row.append(val)
                else:
                    submission_row.append(0.0)
            
            writer.writerow(submission_row)
    
    print(f"✅ Focused submission saved to {filename}")
    
    # Statistics
    flat_preds = [val for row in predictions for val in row]
    if flat_preds:
        pred_mean = sum(flat_preds) / len(flat_preds)
        pred_std = math.sqrt(sum((x - pred_mean) ** 2 for x in flat_preds) / len(flat_preds))
        print(f"📊 Predictions: mean={pred_mean:.4f}, std={pred_std:.4f}")
        print(f"📊 Range: [{min(flat_preds):.4f}, {max(flat_preds):.4f}]")

def main():
    """Main execution function"""
    print("🎯 Shell.ai Hackathon 2025 - FOCUSED Improvement Solution")
    print("=" * 65)
    print("🔄 Building on Score 22 → Target: Score > 30")
    print("⚡ Focused improvements, avoiding overfitting")
    print()
    
    try:
        # Load data
        print("🔄 Loading data...")
        train_headers, train_data = load_csv('train.csv')
        test_headers, test_data = load_csv('test.csv')
        
        print(f"📊 Data: Train {len(train_data)}×{len(train_headers)}, Test {len(test_data)}×{len(test_headers)}")
        
        # Focused feature engineering
        print("\n🎯 Phase 1: Focused Feature Engineering")
        train_headers_eng, train_data_eng = focused_feature_engineering(train_headers, train_data)
        test_headers_eng, test_data_eng = focused_feature_engineering(test_headers, test_data)
        
        # Focused data preparation
        print("\n🎯 Phase 2: Focused Data Preparation")
        X_train, y_train, X_test, target_names = prepare_data_focused(
            train_headers_eng, train_data_eng, 
            test_headers_eng, test_data_eng
        )
        
        print(f"\n📈 Final dataset: {len(X_train)} train, {len(X_test)} test, {len(X_train[0]) if X_train else 0} features")
        
        # Make predictions with conservative bounds
        print("\n🎯 Phase 3: Improved Ensemble Prediction")
        predictions = predict_with_conservative_bounds(X_train, y_train, X_test, target_names)
        
        # Save submission
        print("\n🎯 Phase 4: Submission Generation")
        save_focused_submission(predictions, test_data, test_headers, target_names)
        
        print("\n" + "="*65)
        print("🎉 FOCUSED IMPROVEMENT PIPELINE COMPLETED!")
        print("📁 Generated: submission.csv")
        print("🏆 Expected: Score > 22 (building on what worked)")
        print("🎯 Focused improvements without overfitting")
        print("="*65)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()