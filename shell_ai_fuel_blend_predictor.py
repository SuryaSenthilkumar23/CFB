#!/usr/bin/env python3
"""
Shell.ai Hackathon Level 1: Fuel Blend Properties Prediction
Elite ML Pipeline for predicting 10 blend properties from component data

Author: AI Assistant (a16z-backed startup style)
Goal: Minimize MAPE across 10 target properties
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FuelBlendPredictor:
    """
    Production-grade ML pipeline for fuel blend properties prediction
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        self.cv_scores = {}
        self.best_params = {}
        
    def load_and_analyze_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and perform initial data analysis"""
        print("🔍 Loading and analyzing data...")
        
        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        # Basic statistics
        print("\n📊 Data Quality Check:")
        print(f"Train nulls: {train_df.isnull().sum().sum()}")
        print(f"Test nulls: {test_df.isnull().sum().sum()}")
        
        # Identify feature columns
        component_cols = [col for col in train_df.columns if 'Component' in col and '%' in col]
        property_cols = [col for col in train_df.columns if 'Property' in col and 'Blend' not in col]
        target_cols = [col for col in train_df.columns if 'BlendProperty' in col]
        
        print(f"\n🧪 Feature Structure:")
        print(f"Component % columns: {len(component_cols)}")
        print(f"Component property columns: {len(property_cols)}")
        print(f"Target columns: {len(target_cols)}")
        
        # Check for zero variance features
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        zero_var_cols = []
        for col in numeric_cols:
            if col not in target_cols and train_df[col].var() == 0:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            print(f"⚠️ Zero variance features found: {zero_var_cols}")
        
        return train_df, test_df
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Smart feature engineering pipeline"""
        print("\n🛠️ Engineering features...")
        
        # Combine datasets for consistent feature engineering
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        train_size = len(train_df)
        
        # Identify column types
        component_pct_cols = [col for col in combined_df.columns if 'Component' in col and '%' in col]
        property_cols = [col for col in combined_df.columns if 'Property' in col and 'Blend' not in col]
        target_cols = [col for col in combined_df.columns if 'BlendProperty' in col]
        
        # 1. Normalize component percentages (should sum to 100%)
        if component_pct_cols:
            combined_df['ComponentSum'] = combined_df[component_pct_cols].sum(axis=1)
            for col in component_pct_cols:
                combined_df[f'{col}_normalized'] = combined_df[col] / combined_df['ComponentSum']
        
        # 2. Component property aggregations
        if property_cols:
            # Group by component (assuming naming convention like Component1_Property1, etc.)
            component_groups = {}
            for col in property_cols:
                component = col.split('_')[0] if '_' in col else col.split('Property')[0] + 'Property'
                if component not in component_groups:
                    component_groups[component] = []
                component_groups[component].append(col)
            
            # Create aggregated features per component
            for component, props in component_groups.items():
                if len(props) > 1:
                    combined_df[f'{component}_mean'] = combined_df[props].mean(axis=1)
                    combined_df[f'{component}_std'] = combined_df[props].std(axis=1)
                    combined_df[f'{component}_min'] = combined_df[props].min(axis=1)
                    combined_df[f'{component}_max'] = combined_df[props].max(axis=1)
        
        # 3. Component interactions
        if len(component_pct_cols) >= 2:
            for i, col1 in enumerate(component_pct_cols):
                for col2 in component_pct_cols[i+1:]:
                    combined_df[f'{col1}_x_{col2}'] = combined_df[col1] * combined_df[col2]
                    if combined_df[col2].sum() != 0:
                        combined_df[f'{col1}_div_{col2}'] = combined_df[col1] / (combined_df[col2] + 1e-8)
        
        # 4. Weighted property features (component % * property values)
        if component_pct_cols and property_cols:
            for pct_col in component_pct_cols:
                component_name = pct_col.replace('%', '').replace('Component', '')
                matching_props = [col for col in property_cols if component_name in col]
                for prop_col in matching_props:
                    combined_df[f'{pct_col}_weighted_{prop_col}'] = combined_df[pct_col] * combined_df[prop_col]
        
        # 5. PCA features for dimensionality reduction
        if len(property_cols) > 10:
            pca = PCA(n_components=min(10, len(property_cols)), random_state=self.random_state)
            pca_features = pca.fit_transform(combined_df[property_cols].fillna(0))
            for i in range(pca_features.shape[1]):
                combined_df[f'PCA_Property_{i+1}'] = pca_features[:, i]
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Remove zero variance features
        feature_cols = [col for col in combined_df.columns if col not in target_cols]
        zero_var_cols = []
        for col in feature_cols:
            if combined_df[col].var() == 0:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            print(f"Removing {len(zero_var_cols)} zero variance features")
            combined_df = combined_df.drop(columns=zero_var_cols)
        
        # Split back
        train_engineered = combined_df[:train_size].copy()
        test_engineered = combined_df[train_size:].copy()
        
        # Remove target columns from test set
        test_cols = [col for col in test_engineered.columns if col not in target_cols]
        test_engineered = test_engineered[test_cols]
        
        print(f"✅ Feature engineering complete. New shape: {train_engineered.shape}")
        return train_engineered, test_engineered
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_type: str, n_trials: int = 100) -> Dict:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': self.random_state
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': self.random_state,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                    'random_state': self.random_state,
                    'verbose': False
                }
                model = cb.CatBoostRegressor(**params)
            
            # Cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_percentage_error')
            return -scores.mean()  # Return positive MAPE
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                            optimize_params: bool = True) -> None:
        """Train ensemble models for each target"""
        print("\n🚀 Training ensemble models...")
        
        target_cols = y_train.columns.tolist()
        
        for target in target_cols:
            print(f"\n🎯 Training models for {target}")
            y = y_train[target]
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            self.scalers[target] = scaler
            
            models = {}
            
            # XGBoost
            if optimize_params:
                print("Optimizing XGBoost hyperparameters...")
                xgb_params = self.optimize_hyperparameters(X_scaled, y, 'xgboost', n_trials=50)
                self.best_params[f'{target}_xgboost'] = xgb_params
            else:
                xgb_params = {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': self.random_state}
            
            models['xgboost'] = xgb.XGBRegressor(**xgb_params)
            
            # LightGBM
            if optimize_params:
                print("Optimizing LightGBM hyperparameters...")
                lgb_params = self.optimize_hyperparameters(X_scaled, y, 'lightgbm', n_trials=50)
                self.best_params[f'{target}_lightgbm'] = lgb_params
            else:
                lgb_params = {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': self.random_state, 'verbose': -1}
            
            models['lightgbm'] = lgb.LGBMRegressor(**lgb_params)
            
            # CatBoost
            if optimize_params:
                print("Optimizing CatBoost hyperparameters...")
                cb_params = self.optimize_hyperparameters(X_scaled, y, 'catboost', n_trials=50)
                self.best_params[f'{target}_catboost'] = cb_params
            else:
                cb_params = {'iterations': 500, 'depth': 6, 'learning_rate': 0.1, 'random_state': self.random_state, 'verbose': False}
            
            models['catboost'] = cb.CatBoostRegressor(**cb_params)
            
            # Train individual models and collect feature importance
            trained_models = []
            for name, model in models.items():
                print(f"Training {name}...")
                model.fit(X_scaled, y)
                trained_models.append((name, model))
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X_scaled.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[f'{target}_{name}'] = importance_df
            
            # Create ensemble with equal weights (can be optimized further)
            ensemble = VotingRegressor(trained_models)
            ensemble.fit(X_scaled, y)
            
            self.models[target] = ensemble
            
            # Cross-validation score
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(ensemble, X_scaled, y, cv=kf, scoring='neg_mean_absolute_percentage_error')
            self.cv_scores[target] = -cv_scores.mean()
            
            print(f"✅ {target} CV MAPE: {self.cv_scores[target]:.4f}")
    
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for test set"""
        print("\n🔮 Generating predictions...")
        
        predictions = {}
        
        for target, model in self.models.items():
            # Scale features using the same scaler as training
            X_scaled = pd.DataFrame(
                self.scalers[target].transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
            
            pred = model.predict(X_scaled)
            predictions[target] = pred
        
        pred_df = pd.DataFrame(predictions, index=X_test.index)
        print(f"✅ Predictions generated for {len(pred_df)} samples")
        
        return pred_df
    
    def evaluate_train_performance(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict:
        """Evaluate performance on training set"""
        print("\n📈 Evaluating training performance...")
        
        train_predictions = {}
        mape_scores = {}
        
        for target in y_train.columns:
            # Scale features
            X_scaled = pd.DataFrame(
                self.scalers[target].transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            
            pred = self.models[target].predict(X_scaled)
            train_predictions[target] = pred
            
            # Calculate MAPE
            mape = mean_absolute_percentage_error(y_train[target], pred)
            mape_scores[target] = mape
        
        avg_mape = np.mean(list(mape_scores.values()))
        mape_scores['Average'] = avg_mape
        
        print(f"\n🎯 Training MAPE Scores:")
        for target, score in mape_scores.items():
            print(f"{target}: {score:.4f}")
        
        return mape_scores, pd.DataFrame(train_predictions, index=X_train.index)
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance for each model"""
        print(f"\n📊 Plotting top {top_n} feature importance...")
        
        # Create subplots
        n_targets = len([k for k in self.feature_importance.keys() if 'xgboost' in k])
        fig, axes = plt.subplots(n_targets, 3, figsize=(20, 6*n_targets))
        
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        
        target_idx = 0
        for key, importance_df in self.feature_importance.items():
            if 'xgboost' in key:
                target = key.replace('_xgboost', '')
                
                # XGBoost
                top_features = importance_df.head(top_n)
                axes[target_idx, 0].barh(range(len(top_features)), top_features['importance'])
                axes[target_idx, 0].set_yticks(range(len(top_features)))
                axes[target_idx, 0].set_yticklabels(top_features['feature'])
                axes[target_idx, 0].set_title(f'{target} - XGBoost')
                axes[target_idx, 0].invert_yaxis()
                
                # LightGBM
                lgb_key = key.replace('xgboost', 'lightgbm')
                if lgb_key in self.feature_importance:
                    lgb_importance = self.feature_importance[lgb_key].head(top_n)
                    axes[target_idx, 1].barh(range(len(lgb_importance)), lgb_importance['importance'])
                    axes[target_idx, 1].set_yticks(range(len(lgb_importance)))
                    axes[target_idx, 1].set_yticklabels(lgb_importance['feature'])
                    axes[target_idx, 1].set_title(f'{target} - LightGBM')
                    axes[target_idx, 1].invert_yaxis()
                
                # CatBoost
                cb_key = key.replace('xgboost', 'catboost')
                if cb_key in self.feature_importance:
                    cb_importance = self.feature_importance[cb_key].head(top_n)
                    axes[target_idx, 2].barh(range(len(cb_importance)), cb_importance['importance'])
                    axes[target_idx, 2].set_yticks(range(len(cb_importance)))
                    axes[target_idx, 2].set_yticklabels(cb_importance['feature'])
                    axes[target_idx, 2].set_title(f'{target} - CatBoost')
                    axes[target_idx, 2].invert_yaxis()
                
                target_idx += 1
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main execution pipeline"""
    print("🏆 Shell.ai Hackathon Level 1 - Fuel Blend Properties Prediction")
    print("=" * 60)
    
    # Initialize predictor
    predictor = FuelBlendPredictor(random_state=42)
    
    # File paths (adjust as needed)
    train_path = 'train.csv'
    test_path = 'test.csv'
    sample_solution_path = 'sample_solution.csv'
    
    try:
        # 1. Load and analyze data
        train_df, test_df = predictor.load_and_analyze_data(train_path, test_path)
        
        # 2. Feature engineering
        train_engineered, test_engineered = predictor.engineer_features(train_df, test_df)
        
        # 3. Prepare features and targets
        target_cols = [col for col in train_engineered.columns if 'BlendProperty' in col]
        feature_cols = [col for col in train_engineered.columns if col not in target_cols]
        
        X_train = train_engineered[feature_cols]
        y_train = train_engineered[target_cols]
        X_test = test_engineered[feature_cols]
        
        print(f"\n🎯 Final dataset shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        
        # 4. Train models (set optimize_params=True for full optimization)
        predictor.train_ensemble_models(X_train, y_train, optimize_params=False)  # Set to True for production
        
        # 5. Evaluate training performance
        train_mape, train_preds = predictor.evaluate_train_performance(X_train, y_train)
        
        # 6. Generate test predictions
        test_predictions = predictor.predict(X_test)
        
        # 7. Create submission file
        submission_df = test_predictions.copy()
        if 'ID' in test_df.columns:
            submission_df.insert(0, 'ID', test_df['ID'])
        elif test_df.index.name:
            submission_df.insert(0, 'ID', test_df.index)
        else:
            submission_df.insert(0, 'ID', range(len(submission_df)))
        
        submission_df.to_csv('submission.csv', index=False)
        print(f"\n✅ Submission file saved: submission.csv")
        
        # 8. Plot feature importance
        predictor.plot_feature_importance(top_n=15)
        
        # 9. Save model insights
        insights = {
            'cv_scores': predictor.cv_scores,
            'train_mape': train_mape,
            'best_params': predictor.best_params,
            'avg_cv_mape': np.mean(list(predictor.cv_scores.values())),
            'avg_train_mape': train_mape['Average']
        }
        
        print(f"\n🎯 Final Results Summary:")
        print(f"Average CV MAPE: {insights['avg_cv_mape']:.4f}")
        print(f"Average Train MAPE: {insights['avg_train_mape']:.4f}")
        
        # Save insights
        import json
        with open('model_insights.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print(f"\n🚀 Pipeline completed successfully!")
        print(f"📁 Files generated:")
        print(f"   - submission.csv")
        print(f"   - feature_importance.png")
        print(f"   - model_insights.json")
        
        # Future improvement suggestions
        print(f"\n💡 Future Optimization Ideas:")
        print(f"   1. Hyperparameter tuning with more trials")
        print(f"   2. Advanced ensembling (stacking, blending)")
        print(f"   3. Neural networks for complex interactions")
        print(f"   4. Graph Neural Networks for component relationships")
        print(f"   5. Mixture models for different fuel types")
        print(f"   6. Time-series features if temporal data available")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print(f"Please ensure train.csv, test.csv are in the current directory")
        
        # Create sample data structure for testing
        print(f"\n🔧 Creating sample data structure for testing...")
        create_sample_data()
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


def create_sample_data():
    """Create sample data for testing the pipeline"""
    np.random.seed(42)
    
    # Sample data structure based on description
    n_train = 1000
    n_test = 300
    
    # Component percentages (5 components)
    component_cols = [f'%Component{i+1}' for i in range(5)]
    
    # Component properties (10 per component = 50 total)
    property_cols = []
    for i in range(5):
        for j in range(10):
            property_cols.append(f'Component{i+1}_Property{j+1}')
    
    # Target columns
    target_cols = [f'BlendProperty{i+1}' for i in range(10)]
    
    # Generate training data
    train_data = {}
    
    # Component percentages (sum to ~100)
    comp_base = np.random.dirichlet(np.ones(5), n_train) * 100
    for i, col in enumerate(component_cols):
        train_data[col] = comp_base[:, i]
    
    # Component properties
    for col in property_cols:
        train_data[col] = np.random.normal(50, 15, n_train)
    
    # Generate correlated targets (simplified)
    for i, col in enumerate(target_cols):
        # Create some correlation with components and properties
        target_val = (
            0.3 * train_data[component_cols[i % 5]] +
            0.2 * train_data[property_cols[i * 5]] +
            0.1 * train_data[property_cols[i * 5 + 1]] +
            np.random.normal(0, 5, n_train)
        )
        train_data[col] = np.maximum(target_val, 0.1)  # Ensure positive values
    
    train_df = pd.DataFrame(train_data)
    
    # Generate test data (same structure, no targets)
    test_data = {}
    comp_base_test = np.random.dirichlet(np.ones(5), n_test) * 100
    for i, col in enumerate(component_cols):
        test_data[col] = comp_base_test[:, i]
    
    for col in property_cols:
        test_data[col] = np.random.normal(50, 15, n_test)
    
    test_df = pd.DataFrame(test_data)
    
    # Save sample data
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    # Create sample solution format
    sample_solution = pd.DataFrame({
        'ID': range(n_test),
        **{col: np.random.uniform(10, 100, n_test) for col in target_cols}
    })
    sample_solution.to_csv('sample_solution.csv', index=False)
    
    print(f"✅ Sample data created:")
    print(f"   - train.csv: {train_df.shape}")
    print(f"   - test.csv: {test_df.shape}")
    print(f"   - sample_solution.csv: {sample_solution.shape}")


if __name__ == "__main__":
    main()