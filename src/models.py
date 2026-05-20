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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
                "max_depth":         [5, 10],
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
                "n_estimators":    [100, 200],
                "max_depth":       [3, 5],
                "learning_rate":   [0.05, 0.1],
                "min_child_weight":[1, 5],
                "subsample":       [0.8, 1.0],
            },
            "scaled":            False,
            "encoded":           True,
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
            ),
            "params": {
                "n_estimators":     [100, 200],
                "max_depth":        [3, 5],
                "learning_rate":    [0.05, 0.1],
                "min_child_samples":[20, 50],
                "num_leaves":       [31, 63],
            },
            "scaled":            False,
            "encoded":           True,
            "use_sample_weight": True,
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
        {
            # Soft-voting ensemble: RF (tree, Sideways-strong) + LGB (gradient
            # boosting, Bear-capable) — deliberately diverse prediction profiles.
            # RF uses unscaled features; LGB also works on raw features.
            "name": "Voting Ensemble",
            "estimator": VotingClassifier(
                estimators=[
                    ("rf",  RandomForestClassifier(
                        n_estimators=100, max_depth=10,
                        class_weight="balanced", random_state=42, n_jobs=-1)),
                    ("lgb", lgb.LGBMClassifier(
                        objective="multiclass", num_class=3,
                        n_estimators=200, max_depth=5, learning_rate=0.1,
                        min_child_samples=50, num_leaves=31,
                        random_state=42, n_jobs=-1, verbose=-1)),
                ],
                voting="soft",
                n_jobs=-1,
            ),
            "params": {},
            "scaled":  False,
            "encoded": False,
            "use_sample_weight": True,
        },
    ]
