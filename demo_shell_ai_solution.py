#!/usr/bin/env python3
"""
Shell.ai Hackathon Level 1 - Demo Solution
Simplified version showing the complete ML pipeline architecture
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple

print("🏆 Shell.ai Hackathon Level 1 - Fuel Blend Properties Prediction")
print("=" * 60)

class SimpleFuelBlendPredictor:
    """
    Simplified demo version of the ML pipeline
    Shows the complete architecture without external dependencies
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def create_sample_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sample data matching the hackathon structure"""
        print("🔧 Creating sample data structure...")
        
        n_train = 1000
        n_test = 300
        
        # 5 component percentages + 50 component properties = 55 features
        n_features = 55
        n_targets = 10
        
        # Generate training data
        X_train = np.random.randn(n_train, n_features)
        
        # Component percentages (first 5 features) - ensure they're positive and sum to ~100
        component_percentages = np.random.dirichlet(np.ones(5), n_train) * 100
        X_train[:, :5] = component_percentages
        
        # Component properties (remaining 50 features)
        X_train[:, 5:] = np.random.normal(50, 15, (n_train, 50))
        
        # Generate correlated targets
        y_train = np.zeros((n_train, n_targets))
        for i in range(n_targets):
            # Create correlation with components and properties
            y_train[:, i] = (
                0.3 * X_train[:, i % 5] +  # Component correlation
                0.2 * X_train[:, 5 + i * 5] +  # Property correlation
                0.1 * X_train[:, 10 + i * 3] +  # Additional property
                np.random.normal(0, 5, n_train)  # Noise
            )
            y_train[:, i] = np.maximum(y_train[:, i], 0.1)  # Ensure positive
        
        # Generate test data
        X_test = np.random.randn(n_test, n_features)
        component_percentages_test = np.random.dirichlet(np.ones(5), n_test) * 100
        X_test[:, :5] = component_percentages_test
        X_test[:, 5:] = np.random.normal(50, 15, (n_test, 50))
        
        print(f"✅ Sample data created:")
        print(f"   - X_train: {X_train.shape}")
        print(f"   - y_train: {y_train.shape}")
        print(f"   - X_test: {X_test.shape}")
        
        return X_train, y_train, X_test
    
    def engineer_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Feature engineering pipeline"""
        print("\n🛠️ Engineering features...")
        
        # Combine for consistent feature engineering
        X_combined = np.vstack([X_train, X_test])
        train_size = X_train.shape[0]
        
        features = []
        feature_names = []
        
        # Original features
        features.append(X_combined)
        feature_names.extend([f'Feature_{i}' for i in range(X_combined.shape[1])])
        
        # Component percentage normalization (first 5 features)
        component_sum = X_combined[:, :5].sum(axis=1, keepdims=True)
        normalized_components = X_combined[:, :5] / (component_sum + 1e-8)
        features.append(normalized_components)
        feature_names.extend([f'Component_{i}_normalized' for i in range(5)])
        
        # Component interactions (first 5 features)
        for i in range(5):
            for j in range(i + 1, 5):
                interaction = X_combined[:, i] * X_combined[:, j]
                features.append(interaction.reshape(-1, 1))
                feature_names.append(f'Component_{i}_x_{j}')
                
                ratio = X_combined[:, i] / (X_combined[:, j] + 1e-8)
                features.append(ratio.reshape(-1, 1))
                feature_names.append(f'Component_{i}_div_{j}')
        
        # Property aggregations (features 5-54, grouped by component)
        for comp in range(5):
            start_idx = 5 + comp * 10
            end_idx = start_idx + 10
            comp_properties = X_combined[:, start_idx:end_idx]
            
            # Aggregated statistics
            features.append(comp_properties.mean(axis=1, keepdims=True))
            feature_names.append(f'Component_{comp}_prop_mean')
            
            features.append(comp_properties.std(axis=1, keepdims=True))
            feature_names.append(f'Component_{comp}_prop_std')
            
            features.append(comp_properties.min(axis=1, keepdims=True))
            feature_names.append(f'Component_{comp}_prop_min')
            
            features.append(comp_properties.max(axis=1, keepdims=True))
            feature_names.append(f'Component_{comp}_prop_max')
        
        # Weighted features (component % * properties)
        for comp in range(5):
            comp_pct = X_combined[:, comp]
            start_idx = 5 + comp * 10
            end_idx = start_idx + 10
            comp_properties = X_combined[:, start_idx:end_idx]
            
            weighted = comp_pct.reshape(-1, 1) * comp_properties
            features.append(weighted)
            feature_names.extend([f'Weighted_{comp}_{i}' for i in range(10)])
        
        # Combine all features
        X_engineered = np.hstack(features)
        
        # Remove zero variance features
        variances = np.var(X_engineered, axis=0)
        non_zero_var_mask = variances > 1e-8
        X_engineered = X_engineered[:, non_zero_var_mask]
        feature_names = [name for i, name in enumerate(feature_names) if non_zero_var_mask[i]]
        
        # Split back
        X_train_eng = X_engineered[:train_size]
        X_test_eng = X_engineered[train_size:]
        
        print(f"✅ Feature engineering complete:")
        print(f"   - Original features: {X_train.shape[1]}")
        print(f"   - Engineered features: {X_train_eng.shape[1]}")
        print(f"   - Features removed (zero variance): {sum(~non_zero_var_mask)}")
        
        return X_train_eng, X_test_eng
    
    def simple_ensemble_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """Simple ensemble using linear models (demo version)"""
        print("\n🚀 Training simple ensemble models...")
        
        n_targets = y_train.shape[1]
        predictions = np.zeros((X_test.shape[0], n_targets))
        
        # Standardize features
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_train_scaled = (X_train - X_mean) / X_std
        X_test_scaled = (X_test - X_mean) / X_std
        
        for target_idx in range(n_targets):
            print(f"Training model for target {target_idx + 1}/10")
            
            y_target = y_train[:, target_idx]
            
            # Simple linear regression with regularization (Ridge-like)
            # X_train_scaled @ w = y_target
            # w = (X^T X + lambda I)^-1 X^T y
            lambda_reg = 0.01
            XtX = X_train_scaled.T @ X_train_scaled
            XtX += lambda_reg * np.eye(XtX.shape[0])
            Xty = X_train_scaled.T @ y_target
            
            try:
                weights = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                weights = np.linalg.pinv(X_train_scaled) @ y_target
            
            # Predict
            pred = X_test_scaled @ weights
            predictions[:, target_idx] = pred
            
            # Calculate simple feature importance (absolute weights)
            importance = np.abs(weights)
            top_features = np.argsort(importance)[-10:][::-1]
            self.feature_importance[f'Target_{target_idx}'] = {
                'top_features': top_features.tolist(),
                'importance_scores': importance[top_features].tolist()
            }
        
        print("✅ Ensemble training complete!")
        return predictions
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        
        # 1. Create sample data
        X_train, y_train, X_test = self.create_sample_data()
        
        # 2. Feature engineering
        X_train_eng, X_test_eng = self.engineer_features(X_train, X_test)
        
        # 3. Train models and predict
        predictions = self.simple_ensemble_predict(X_train_eng, y_train, X_test_eng)
        
        # 4. Evaluate on training set (for demo)
        train_predictions = self.simple_ensemble_predict(X_train_eng, y_train, X_train_eng)
        
        # Calculate MAPE for each target
        mape_scores = {}
        print(f"\n📈 Training Performance (MAPE):")
        for i in range(y_train.shape[1]):
            mape = self.calculate_mape(y_train[:, i], train_predictions[:, i])
            mape_scores[f'BlendProperty{i+1}'] = mape / 100  # Convert to decimal
            print(f"BlendProperty{i+1}: {mape:.4f}%")
        
        avg_mape = np.mean(list(mape_scores.values()))
        mape_scores['Average'] = avg_mape
        print(f"Average MAPE: {avg_mape*100:.4f}%")
        
        # 5. Create submission file
        submission_data = {
            'ID': list(range(len(predictions))),
        }
        for i in range(predictions.shape[1]):
            submission_data[f'BlendProperty{i+1}'] = predictions[:, i].tolist()
        
        # Save as simple format (would be CSV in real implementation)
        print(f"\n✅ Saving results...")
        
        # Save submission
        with open('submission_demo.json', 'w') as f:
            json.dump(submission_data, f, indent=2)
        
        # Save insights
        insights = {
            'performance': mape_scores,
            'feature_importance': self.feature_importance,
            'model_info': {
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'original_features': X_train.shape[1],
                'engineered_features': X_train_eng.shape[1],
                'targets': y_train.shape[1]
            }
        }
        
        with open('model_insights_demo.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"📁 Files created:")
        print(f"   - submission_demo.json")
        print(f"   - model_insights_demo.json")
        
        # 6. Display sample predictions
        print(f"\n🔮 Sample Predictions:")
        print(f"{'ID':<5} {'BlendProp1':<12} {'BlendProp2':<12} {'BlendProp3':<12}")
        print("-" * 45)
        for i in range(min(5, len(predictions))):
            print(f"{i:<5} {predictions[i, 0]:<12.4f} {predictions[i, 1]:<12.4f} {predictions[i, 2]:<12.4f}")
        
        return insights


def main():
    """Main execution"""
    predictor = SimpleFuelBlendPredictor(random_state=42)
    insights = predictor.run_pipeline()
    
    print(f"\n🎯 Pipeline Summary:")
    print(f"   Average MAPE: {insights['performance']['Average']*100:.4f}%")
    print(f"   Features: {insights['model_info']['engineered_features']}")
    print(f"   Targets: {insights['model_info']['targets']}")
    
    print(f"\n💡 Next Steps for Real Implementation:")
    print(f"   1. Replace this demo with the full pipeline (shell_ai_fuel_blend_predictor.py)")
    print(f"   2. Install required packages: pip install -r requirements.txt")
    print(f"   3. Place real train.csv and test.csv files in directory")
    print(f"   4. Run: python shell_ai_fuel_blend_predictor.py")
    print(f"   5. Enable hyperparameter optimization for better performance")
    
    print(f"\n🏆 Ready for Shell.ai Hackathon! Good luck! 🚀")


if __name__ == "__main__":
    main()