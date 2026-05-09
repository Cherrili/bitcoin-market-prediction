"""
Step 5 — Model definitions and hyperparameter search grids.

Public API
----------
get_model_configs() -> list[dict]

Each dict has keys:
    name      : str
    estimator : sklearn-compatible estimator
    params    : dict  (hyperparameter grid for GridSearchCV)
    scaled    : bool  (True → use StandardScaler output)
    encoded   : bool  (True → labels must be in {0,1,2} for XGB/LGB)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb


def get_model_configs() -> list:
    return [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                solver="lbfgs", max_iter=2000, random_state=42,
                class_weight="balanced",
            ),
            "params":  {"C": [0.01, 0.1, 1, 10]},
            "scaled":  True,
            "encoded": False,
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1,
                                                class_weight="balanced"),
            "params": {
                "n_estimators":      [100, 200],
                "max_depth":         [5, 10, None],
                "min_samples_split": [2, 5],
            },
            "scaled":  False,
            "encoded": False,
        },
        {
            "name": "XGBoost",
            "estimator": xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators":  [100, 200],
                "max_depth":     [3, 5],
                "learning_rate": [0.05, 0.1],
            },
            "scaled":           False,
            "encoded":          True,
            "use_sample_weight": True,
        },
        {
            "name": "LightGBM",
            "estimator": lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                class_weight="balanced",
            ),
            "params": {
                "n_estimators":  [100, 200],
                "max_depth":     [3, 5, -1],
                "learning_rate": [0.05, 0.1],
            },
            "scaled":  False,
            "encoded": True,
        },
        {
            "name": "SVM",
            "estimator": SVC(probability=True, random_state=42,
                             class_weight="balanced"),
            "params": {
                "C":      [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
            },
            "scaled":  True,
            "encoded": False,
        },
        {
            "name": "KNN",
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "params": {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights":     ["uniform", "distance"],
            },
            "scaled":  True,
            "encoded": False,
        },
    ]
