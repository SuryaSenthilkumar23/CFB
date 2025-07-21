import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ML libraries
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

import optuna

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

############################################################
# Utility functions
############################################################

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Percentage Error (MAPE)."""
    epsilon = 1e-9  # to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))

# scorer for sklearn
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


def identify_target_columns(df: pd.DataFrame) -> List[str]:
    """Automatically detect target columns (BlendProperty1..10)."""
    target_cols = [col for col in df.columns if col.lower().startswith("blendproperty")]
    if not target_cols:
        # fallback: any numeric column not ending with '%' but maybe by index position (last 10 cols)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_cols = numeric_cols[-10:].tolist()
    return target_cols


def identify_percentage_columns(df: pd.DataFrame) -> List[str]:
    """Detect percentage columns using simple heuristics."""
    patterns = ["%", "percent", "_pct", "_percentage"]
    pct_cols = [col for col in df.columns if any(pat.lower() in col.lower() for pat in patterns)]
    # Additional heuristic: first 5 numeric columns summing roughly to 100
    if len(pct_cols) < 5:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        subset = numeric_cols[:5]
        # check if sums around 100 ± 5
        if np.isclose(df[subset].sum(axis=1).median(), 100, atol=5):
            pct_cols = subset.tolist()
    return pct_cols


def component_groups(df: pd.DataFrame, n_components: int = 5) -> Dict[int, List[str]]:
    """Group columns by component index (assumes 'Component{i}' in name)."""
    groups = {}
    for i in range(1, n_components + 1):
        prefix = f"Component{i}"
        groups[i] = [col for col in df.columns if prefix.lower() in col.lower()]
    return groups


############################################################
# Feature Engineering
############################################################

def feature_engineering(df: pd.DataFrame, pct_cols: List[str]) -> pd.DataFrame:
    """Perform feature engineering on raw dataframe and return processed features."""
    df_feat = df.copy()

    # Fill missing values with column median
    df_feat = df_feat.fillna(df_feat.median())

    # Normalize percentage columns to sum to 1
    if pct_cols:
        pct_sum = df_feat[pct_cols].sum(axis=1).replace(0, 1)  # avoid division by zero
        df_feat[pct_cols] = df_feat[pct_cols].div(pct_sum, axis=0)

    # Aggregate stats per component (mean, std)
    comp_groups = component_groups(df_feat)
    for comp_idx, cols in comp_groups.items():
        if len(cols) > 0:
            df_feat[f"Component{comp_idx}_mean"] = df_feat[cols].mean(axis=1)
            df_feat[f"Component{comp_idx}_std"] = df_feat[cols].std(axis=1)

    # Ratio features between percentage and aggregate mean
    for col in pct_cols:
        comp_idx = None
        for i in range(1, 6):
            if f"Component{i}".lower() in col.lower():
                comp_idx = i
                break
        if comp_idx:
            df_feat[f"{col}_to_mean"] = df_feat[col] / (df_feat[f"Component{comp_idx}_mean"] + 1e-6)

    # Variance threshold to remove zero/near-zero variance features
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(df_feat)
    df_feat = df_feat.iloc[:, selector.get_support(indices=True)]

    return df_feat

############################################################
# Model Training & Evaluation
############################################################

def get_models(params_dict: Dict[str, Dict]) -> Dict[str, object]:
    """Instantiate base models with provided parameter dict."""
    return {
        "xgb": XGBRegressor(**params_dict.get("xgb", {}), random_state=RANDOM_STATE, n_jobs=-1),
        "lgbm": LGBMRegressor(**params_dict.get("lgbm", {}), random_state=RANDOM_STATE, n_jobs=-1),
        "cat": CatBoostRegressor(**params_dict.get("cat", {}), random_state=RANDOM_STATE, verbose=False)
    }


def optuna_objective(trial, X: np.ndarray, y: np.ndarray, model_name: str):
    """Optuna objective for hyperparameter tuning."""
    if model_name == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5)
        }
        model = XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10)
        }
        model = LGBMRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    else:  # catboost
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1200),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0)
        }
        model = CatBoostRegressor(**params, random_state=RANDOM_STATE, verbose=False)

    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, valid_idx in cv.split(X):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if model_name == "cat":
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        else:
            model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        scores.append(mean_absolute_percentage_error(y_valid, preds))

    return np.mean(scores)


def tune_hyperparameters(X: np.ndarray, y: np.ndarray, model_name: str, n_trials: int = 20) -> Dict:
    """Tune hyperparameters using Optuna and return best params."""
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective(trial, X, y, model_name), n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_and_predict(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, n_targets: int, n_splits: int = 5) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """Train ensemble models per target and return test predictions and cv metrics."""
    test_preds = np.zeros((X_test.shape[0], n_targets))
    cv_mape: Dict[str, List[float]] = {f"BlendProperty{i+1}": [] for i in range(n_targets)}

    # Hyperparameter tuning once per model (use first target as proxy)
    print("\n[INFO] Hyperparameter tuning for base models (using first target as proxy)...")
    tuned_params = {}
    for model_name in ["xgb", "lgbm", "cat"]:
        tuned_params[model_name] = tune_hyperparameters(X, y[:, 0], model_name, n_trials=15)
        print(f"Best params for {model_name}: {tuned_params[model_name]}")

    base_models = get_models(tuned_params)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for target_idx in range(n_targets):
        print(f"\n[INFO] Training for target BlendProperty{target_idx+1}")
        y_target = y[:, target_idx]

        # store per model predictions for test to ensemble
        test_preds_models = {name: np.zeros(X_test.shape[0]) for name in base_models}

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y_target[train_idx], y_target[valid_idx]

            for model_name, model in base_models.items():
                # Refit model per fold to avoid data leakage
                model_clone = model.__class__(**model.get_params())
                if model_name == "cat":
                    model_clone.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                else:
                    model_clone.fit(X_train, y_train)

                preds_valid = model_clone.predict(X_valid)
                mape = mean_absolute_percentage_error(y_valid, preds_valid)
                cv_mape[f"BlendProperty{target_idx+1}"].append(mape)

                # Accumulate test predictions
                test_preds_models[model_name] += model_clone.predict(X_test) / n_splits

        # Ensemble (simple average)
        ensemble_pred = np.mean(list(test_preds_models.values()), axis=0)
        test_preds[:, target_idx] = ensemble_pred

    return test_preds, cv_mape

############################################################
# Plotting utilities
############################################################

def plot_feature_importance(model, feature_names: List[str], top_n: int = 20, title: str = "Feature Importance"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return  # skip
    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
    plt.title(title)
    plt.tight_layout()
    plt.show()

############################################################
# Main execution
############################################################

def main(args):
    train_path = args.train
    test_path = args.test
    output_path = args.output

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: train or test file not found.")
        sys.exit(1)

    print("[INFO] Loading data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target_cols = identify_target_columns(df_train)
    pct_cols = identify_percentage_columns(df_train)

    feature_cols = [col for col in df_train.columns if col not in target_cols]

    print(f"Detected {len(target_cols)} target columns: {target_cols}")
    print(f"Detected {len(pct_cols)} percentage columns: {pct_cols}")

    # Feature engineering
    print("[INFO] Performing feature engineering...")
    df_train_feat = feature_engineering(df_train[feature_cols], pct_cols)
    df_test_feat = feature_engineering(df_test[feature_cols], pct_cols)

    feature_names = df_train_feat.columns.tolist()

    X = df_train_feat.values
    X_test = df_test_feat.values
    y = df_train[target_cols].values

    # Train models and get predictions
    print("[INFO] Training models and generating predictions...")
    test_preds, cv_mape = train_and_predict(X, y, X_test, n_targets=len(target_cols))

    # Evaluate
    avg_mape_per_target = {t: np.mean(mapes) for t, mapes in cv_mape.items()}
    overall_mape = np.mean(list(avg_mape_per_target.values()))

    print("\n=== CV MAPE per Target ===")
    for t, m in avg_mape_per_target.items():
        print(f"{t}: {m:.5f}")
    print(f"Average MAPE: {overall_mape:.5f}")

    # Prepare submission
    print("[INFO] Saving submission file...")
    submission = pd.DataFrame(test_preds, columns=target_cols)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    # Feature importance plots (optional, using last trained model of first target)
    # plot_feature_importance(model, feature_names)

    # Suggest future improvements
    print("\n[Future Ideas]")
    print("- Try stacking/weighted ensembles based on CV performance\n"
          "- Add polynomial interaction features or domain-specific ratios\n"
          "- Experiment with GNNs or transformer models capturing component relationships\n"
          "- Use grouped cross-validation by batch or other stratification keys if available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blend Properties Prediction Pipeline")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test.csv")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path for output submission file")
    args = parser.parse_args()

    main(args)