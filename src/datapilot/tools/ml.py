import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, List
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def train_gradient_boosting(df: pd.DataFrame, target_col: str, task: str = "regression") -> Dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataframe")
        
    df_clean = df.dropna(subset=[target_col]).copy()
    if len(df_clean) < 20:
        raise ValueError("Insufficient data to train ML model")
        
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    
    # Simple categorical handling for HistGradientBoosting
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype('category')
        
    categorical_features = [X.columns.get_loc(c) for c in cat_cols] if cat_cols else None
    
    if task == "regression":
        model = HistGradientBoostingRegressor(categorical_features=categorical_features, max_iter=100)
        scoring = "r2"
    else:
        model = HistGradientBoostingClassifier(categorical_features=categorical_features, max_iter=100)
        scoring = "accuracy"
        
    # CV performance
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(y)//4), scoring=scoring)
    
    # Train final
    model.fit(X, y)
    
    # Feature Importance
    if HAS_SHAP and not cat_cols: # SHAP TreeExplainer often struggles with sklearn's native categorical handling
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            vals = np.abs(shap_values).mean(0)
            importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['feature', 'importance'])
            importance.sort_values(by=['importance'], ascending=False, inplace=True)
            feature_imp = importance.to_dict(orient='records')
        except Exception:
            feature_imp = _fallback_importance(model, X, y)
    else:
        feature_imp = _fallback_importance(model, X, y)
        
    return {
        "task": task,
        "n_samples": len(y),
        "features": list(X.columns),
        f"cv_mean_{scoring}": float(np.mean(cv_scores)),
        f"cv_std_{scoring}": float(np.std(cv_scores)),
        "feature_importance": feature_imp
    }

def _fallback_importance(model, X, y):
    r = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    importance = pd.DataFrame(list(zip(X.columns, r.importances_mean)), columns=['feature', 'importance'])
    importance.sort_values(by=['importance'], ascending=False, inplace=True)
    return importance.to_dict(orient='records')
