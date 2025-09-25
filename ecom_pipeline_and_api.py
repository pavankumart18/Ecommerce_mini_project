#----------------------Full working code---------------------#

"""
Feature-selection-then-retrain script.

Workflow:
- Load CSV
- Drop duplicates, fix Weekend/Revenue dtypes
- Feature engineering (Total_Page_Events, Total_Duration, Avg_Page_Duration, Pages_per_Session, Product_Time_x_Count, *_log)
- Numeric correlation analysis (print + heatmap)
- Preprocess and train baseline models (LogisticRegression baseline, RandomForest with RandomizedSearchCV) on ALL features (with proper preprocessing)
- Extract RandomForest impurity-based feature importances (including one-hot categorical columns)
- Correlation-guided pruning of numeric features:
    * compute numeric Pearson corr matrix
    * for each highly correlated pair (|r| > corr_threshold), drop the feature with lower importance (if present)
- Select top_k important features after pruning (these are column names of the one-hot-expanded matrix)
- Retrain LR and RF (and XGB if available) on only selected columns (LogisticRegression uses StandardScaler)
- Print metric comparisons and classification reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

warnings.filterwarnings("ignore")
plt.rcParams["figure.max_open_warning"] = 50

# ---------- Config ----------
DATA_PATH = "online_shoppers_intention.csv"      # change if needed
OUT_PATH = "online_shoppers_cleaned_featured.csv"
HEATMAP_PATH = "correlation_heatmap_numeric.png"
FULL_HEATMAP_PATH = "correlation_heatmap_full.png"
corr_threshold = 0.85   # if |r| > this, treat as high correlation (tunable)
TOP_K = 25              # final number of features to keep (after pruning)
RANDOM_STATE = 42

# ---------- Helpers ----------
def make_onehot_encoder():
    # compatibility for different sklearn versions
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def eval_metrics(y_true, y_pred, y_proba=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if (y_proba is not None) else np.nan
    }

# ---------- Load + Feature Engineering ----------
def load_and_engineer(path=DATA_PATH):
    df = pd.read_csv(path)
    print("Loaded shape:", df.shape)

    # drop exact duplicates
    before_dup = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped duplicates: {before_dup - df.shape[0]} rows removed. New shape: {df.shape}")

    # robust boolean -> int for Weekend/Revenue
    if "Weekend" in df.columns:
        if df["Weekend"].dtype == bool:
            df["Weekend"] = df["Weekend"].astype(int)
        elif df["Weekend"].dtype == object:
            df["Weekend"] = df["Weekend"].map(lambda x: 1 if str(x).lower() in ("1","true","yes") else 0)
    if "Revenue" in df.columns:
        if df["Revenue"].dtype == bool:
            df["Revenue"] = df["Revenue"].astype(int)
        elif df["Revenue"].dtype == object:
            df["Revenue"] = df["Revenue"].map(lambda x: 1 if str(x).lower() in ("1","true","yes") else 0)

    # engineered features
    eps = 1e-6
    df["Total_Page_Events"] = df.get("Administrative", 0) + df.get("Informational", 0) + df.get("ProductRelated", 0) + eps
    df["Total_Duration"] = df.get("Administrative_Duration", 0) + df.get("Informational_Duration", 0) + df.get("ProductRelated_Duration", 0) + eps
    df["Avg_Page_Duration"] = df["Total_Duration"] / df["Total_Page_Events"]
    df["Pages_per_Session"] = df["Total_Page_Events"]
    df["Product_Time_x_Count"] = df.get("ProductRelated_Duration", 0) * df.get("ProductRelated", 0)

    # safe log transforms for skewed positives
    skew_candidates = [
        "Administrative", "Informational", "ProductRelated", "PageValues", "ExitRates", "BounceRates",
        "Total_Duration", "Pages_per_Session"
    ]
    for col in skew_candidates:
        if col in df.columns and (df[col] >= 0).all():
            df[col + "_log"] = np.log1p(df[col].clip(lower=0))

    return df

# ---------- Correlation analysis (numeric only) ----------
def correlation_numeric(df, save_path=HEATMAP_PATH, show_plot=False):
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] == 0:
        print("No numeric columns for correlation.")
        return None
    corr = df_num.corr()
    if "Revenue" in df_num.columns:
        print("\nCorrelation of numeric features with Revenue:")
        print(corr["Revenue"].sort_values(ascending=False))
    # heatmap
    plt.figure(figsize=(12,10))
    sns = __import__("seaborn")
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Numeric feature correlation")
    plt.tight_layout()
    try:
        if show_plot:
            plt.show()
    except Exception:
        pass
    plt.savefig(save_path)
    plt.clf()
    print(f"Saved numeric correlation heatmap to {save_path}")
    return corr

# ---------- Preprocess + baseline training to obtain importances ----------
def train_baseline_and_get_importances(df):
    # define feature lists for preprocessor
    numeric_features = [
        "Administrative","Administrative_Duration","Informational","Informational_Duration",
        "ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","PageValues","SpecialDay",
        "Total_Duration","Avg_Page_Duration","Pages_per_Session","Product_Time_x_Count"
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]
    # add any *_log features
    numeric_features += [c for c in df.columns if c.endswith("_log") and c not in numeric_features]

    categorical_features = ["Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend"]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # ensure dtype
    if "Month" in df.columns:
        df["Month"] = df["Month"].astype(str)
    if "VisitorType" in df.columns:
        df["VisitorType"] = df["VisitorType"].astype(str)
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(int)

    X = df[numeric_features + categorical_features].copy()
    y = df["Revenue"].astype(int)

    # split
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE)

    # preprocessors
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", make_onehot_encoder())])

    preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])

    # pipelines
    pipe_lr = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    pipe_rf = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"))])

    print("\nTraining baseline LogisticRegression (on full feature set)...")
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    y_proba_lr = pipe_lr.predict_proba(X_test)[:,1]
    lr_metrics = eval_metrics(y_test, y_pred_lr, y_proba_lr)
    print("LogisticRegression metrics (full features):", lr_metrics)

    print("\nTraining baseline RandomForest (on full feature set) with small RandomizedSearchCV...")
    param_dist_rf = {"clf__n_estimators":[100,200],"clf__max_depth":[6,10,None],"clf__min_samples_split":[2,5,10]}
    rs_rf = RandomizedSearchCV(pipe_rf, param_dist_rf, n_iter=6, cv=3, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1)
    rs_rf.fit(X_train, y_train)
    best_rf = rs_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:,1]
    rf_metrics = eval_metrics(y_test, y_pred_rf, y_proba_rf)
    print("RandomForest metrics (full features):", rf_metrics)
    print("RandomForest best params:", rs_rf.best_params_)

    # Build feature names after preprocessing
    try:
        ohe = rs_rf.best_estimator_.named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
        ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))
    except Exception:
        # fallback: try to get onehot from preprocessor (if different)
        try:
            ohe = rs_rf.best_estimator_.named_steps['pre'].transformers_[1][1].named_steps['onehot']
            ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))
        except Exception:
            ohe_feature_names = []

    all_feature_names = numeric_features + ohe_feature_names
    rf_clf = rs_rf.best_estimator_.named_steps['clf']
    importances = rf_clf.feature_importances_
    fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

    print("\nTop 30 importances (full model):")
    print(fi.head(30).to_string())

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "fi": fi,
        "baseline_metrics": {"logistic": lr_metrics, "rf": rf_metrics},
        "preprocessor": preprocessor,
        "best_rf_estimator": best_rf,
        "X_train_full": X_train, "X_test_full": X_test, "y_train": y_train, "y_test": y_test
    }

# ---------- Correlation-guided pruning + top-K selection ----------
def prune_by_correlation_and_importance(df, fi_series, numeric_features, corr_threshold=corr_threshold, top_k=TOP_K):
    """
    fi_series: pandas Series indexed by the one-hot expanded feature names (same form as X_dum columns)
    numeric_features: list of numeric column names in original df (not one-hot)
    """
    # 1) Compute numeric correlation matrix (original numeric features)
    df_num = df.select_dtypes(include=[np.number])
    # ensure Revenue present
    if "Revenue" not in df_num.columns:
        raise ValueError("Revenue not present as numeric column for correlation-based pruning.")
    corr = df_num.corr().abs()

    # 2) Identify high-correlation pairs among numeric_features (exclude target)
    high_corr_pairs = []
    for i, a in enumerate(numeric_features):
        if a not in corr.columns:
            continue
        for b in numeric_features[i+1:]:
            if b not in corr.columns:
                continue
            r = corr.loc[a,b]
            if r >= corr_threshold:
                high_corr_pairs.append((a,b,r))

    print(f"\nFound {len(high_corr_pairs)} numeric feature pairs with |r| >= {corr_threshold}")
    # 3) Decide which numeric features to drop among highly correlated pairs using importance:
    #    We need importances for numeric features — they appear in fi_series (since fi index is one-hot names,
    #    numeric features should be present as-is in fi_series)
    drop_numeric = set()
    for a,b,r in high_corr_pairs:
        # get importance (0 if missing)
        imp_a = fi_series[a] if a in fi_series.index else 0.0
        imp_b = fi_series[b] if b in fi_series.index else 0.0
        # drop the lower importance feature
        if imp_a >= imp_b:
            drop = b
        else:
            drop = a
        drop_numeric.add(drop)
        print(f"Pruning numeric pair ({a},{b}) r={r:.3f}: dropping '{drop}' (imp {fi_series.get(drop,0):.6f} vs other {fi_series.get(a if drop==b else b,0):.6f})")

    # 4) Build candidate list from fi_series (ordered by importance), but remove any numeric features flagged for drop
    #    fi_series index currently contains the one-hot expanded column names. Numeric features will appear as names without '_'
    candidate_features = [f for f in fi_series.index if f not in drop_numeric]

    # 5) Take top_k from candidate_features
    selected = candidate_features[:top_k]
    print(f"\nSelected top {len(selected)} features after pruning (top {top_k} requested):")
    print(selected)
    return selected, drop_numeric

# ---------- Retrain on selected columns (one-hot expanded) ----------
def retrain_on_selected(df, selected_cols):
    """
    df: original dataframe (engineered)
    selected_cols: list of column names that correspond to columns in the one-hot expanded dataframe
    """
    # Build one-hot expanded full feature matrix (same as used by quick RF earlier)
    # Use get_dummies to match the feature names used in fi
    # We'll create X_full_dum and then subset
    # Choose same features as prior: numeric + categorical names inferred from df
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_candidates = [c for c in df.columns if c not in numeric_candidates]
    # Better approach: use the numeric and categorical lists used before in baseline
    # but for safety we'll just do get_dummies on entire df minus Revenue
    X_all = df.drop(columns=["Revenue"])
    X_dum = pd.get_dummies(X_all, drop_first=False)
    # drop zero-variance if any
    nunique = X_dum.nunique()
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        X_dum = X_dum.drop(columns=zero_var_cols)

    # Ensure selected columns exist
    sel_existing = [c for c in selected_cols if c in X_dum.columns]
    missing = [c for c in selected_cols if c not in X_dum.columns]
    if missing:
        print("\nWarning: some selected columns were not present in the one-hot expanded matrix and will be skipped:", missing)

    X_selected = X_dum[sel_existing].copy()
    y = df["Revenue"].astype(int)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Standardize for LogisticRegression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LogisticRegression on reduced features
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:,1]
    lr_metrics = eval_metrics(y_test, y_pred_lr, y_proba_lr)
    print("\nLogisticRegression metrics (reduced features):", lr_metrics)

    # Train RandomForest on reduced features
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]
    rf_metrics = eval_metrics(y_test, y_pred_rf, y_proba_rf)
    print("RandomForest metrics (reduced features):", rf_metrics)

    # Try XGBoost if available (optional)
    xgb_metrics = None
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        y_proba_xgb = xgb.predict_proba(X_test)[:,1]
        xgb_metrics = eval_metrics(y_test, y_pred_xgb, y_proba_xgb)
        print("XGBoost metrics (reduced features):", xgb_metrics)
    except Exception:
        print("XGBoost not available in this environment; skipped.")

    print("\nClassification report for RandomForest (reduced features):")
    print(classification_report(y_test, y_pred_rf, digits=4))

    results = {"logistic_reduced": lr_metrics, "rf_reduced": rf_metrics, "xgb_reduced": xgb_metrics}
    return results, {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

# ---------- Main ----------
def main():
    print("Loading & engineering...")
    df = load_and_engineer(DATA_PATH)
    print("\nRunning numeric correlation analysis...")
    corr = correlation_numeric(df, save_path=HEATMAP_PATH, show_plot=False)

    print("\nTraining baseline models and obtaining importances...")
    baseline = train_baseline_and_get_importances(df)
    fi = baseline["fi"]
    numeric_features = baseline["numeric_features"]
    categorical_features = baseline["categorical_features"]

    # correlation-guided pruning + top-K selection
    selected, dropped_numeric = prune_by_correlation_and_importance(df, fi, numeric_features, corr_threshold=corr_threshold, top_k=TOP_K)

    # retrain on selected columns
    print("\nRetraining reduced models on selected columns...")
    results_reduced, data_split = retrain_on_selected(df, selected)

    # Compare baseline RF metrics vs reduced RF
    print("\n--- Summary ---")
    print("Baseline RF (full) metrics:", baseline["baseline_metrics"]["rf"])
    print("Reduced RF metrics:", results_reduced["rf_reduced"])
    print("Baseline LR (full) metrics:", baseline["baseline_metrics"]["logistic"])
    print("Reduced LR metrics:", results_reduced["logistic_reduced"])

    # Save engineered dataset for deliverable
    try:
        df.to_csv(OUT_PATH, index=False)
        print(f"\nSaved engineered dataset to: {OUT_PATH}")
    except Exception as e:
        print("Failed to save engineered dataset:", e)

if __name__ == "__main__":
    main()
