import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report,
    roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path):
    return pd.read_csv(path)


def prepare_and_engineer(df):
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert boolean-like columns robustly
    if "Weekend" in df.columns:
        if df["Weekend"].dtype == bool:
            df["Weekend"] = df["Weekend"].astype(int)
        elif df["Weekend"].dtype == object:
            df["Weekend"] = df["Weekend"].map(lambda x: 1 if str(x).lower() in ("1", "true", "yes") else 0)

    if "Revenue" in df.columns:
        if df["Revenue"].dtype == bool:
            df["Revenue"] = df["Revenue"].astype(int)
        elif df["Revenue"].dtype == object:
            df["Revenue"] = df["Revenue"].map(lambda x: 1 if str(x).lower() in ("1", "true", "yes") else 0)

    eps = 1e-6
    df["Total_Page_Events"] = df.get("Administrative", 0) + df.get("Informational", 0) + df.get("ProductRelated", 0) + eps
    df["Total_Duration"] = (
        df.get("Administrative_Duration", 0) + df.get("Informational_Duration", 0) + df.get("ProductRelated_Duration", 0) + eps
    )
    df["Avg_Page_Duration"] = df["Total_Duration"] / df["Total_Page_Events"]
    df["Pages_per_Session"] = df["Total_Page_Events"]
    df["Product_Time_x_Count"] = df.get("ProductRelated_Duration", 0) * df.get("ProductRelated", 0)

    skew_candidates = [
        "Administrative", "Informational", "ProductRelated", "PageValues", "ExitRates", "BounceRates",
        "Total_Duration", "Pages_per_Session"
    ]
    for col in skew_candidates:
        if col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]) and (df[col] >= 0).all():
                df[col + "_log"] = np.log1p(df[col].clip(lower=0))

    return df

#calculates correlation
def correlation_analysis(df, save_path="correlation_heatmap.png", show_plot=True):
    # select numeric columns only (float, int, bool)
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] == 0:
        print("No numeric columns available for correlation.")
        return

    corr = df_num.corr()


    if "Revenue" in df_num.columns:
        target_corr = corr["Revenue"].sort_values(ascending=False)
        print("\nCorrelation of numeric features with Revenue (descending):")
        print(target_corr)
    else:
        print("\nRevenue column not found among numeric columns; skipping target correlation print.")

    # heatmap plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    try:
        if show_plot:
            plt.show()
    except Exception:
        pass
    plt.savefig(save_path)
    print(f"Saved numeric correlation heatmap to: {save_path}")
    plt.clf()


def build_and_evaluate(df, data_out_path=None, random_state=42):
    target = "Revenue"
    if target not in df.columns:
        raise ValueError("Target column 'Revenue' not present in DataFrame.")

    y = df[target].astype(int)

    numeric_features = [
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
        "SpecialDay", "Total_Duration", "Avg_Page_Duration", "Pages_per_Session", "Product_Time_x_Count"
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]
    numeric_features += [c for c in df.columns if c.endswith("_log") and c not in numeric_features]

    categorical_features = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # converting to categorical data types
    if "Month" in df.columns:
        df["Month"] = df["Month"].astype(str)
    if "VisitorType" in df.columns:
        df["VisitorType"] = df["VisitorType"].astype(str)
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(int)

    X = df[numeric_features + categorical_features].copy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=random_state)

    print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])
    print("Positive class distribution (train/val/test):", y_train.mean(), y_val.mean(), y_test.mean())

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", make_onehot_encoder())])

    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])

    pipe_lr = Pipeline(steps=[("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state))])
    pipe_rf = Pipeline(steps=[("pre", preprocessor), ("clf", RandomForestClassifier(random_state=random_state, class_weight="balanced"))])

    print("\nTraining LogisticRegression...")
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    y_proba_lr = pipe_lr.predict_proba(X_test)[:, 1]

    # saving logistic regression model
    try:
        joblib.dump(pipe_lr, "lr_model.pkl")
        print("Saved LogisticRegression pipeline to 'lr_model.pkl'")
    except Exception as e:
        print("Failed to save LogisticRegression model:", e)

    print("\nTraining RandomForest (RandomizedSearchCV)...")
    param_dist_rf = {"clf__n_estimators": [100, 200], "clf__max_depth": [6, 10, None], "clf__min_samples_split": [2, 5, 10]}
    rs_rf = RandomizedSearchCV(pipe_rf, param_dist_rf, n_iter=6, cv=3, scoring="roc_auc", random_state=random_state, n_jobs=-1)
    rs_rf.fit(X_train, y_train)
    best_rf = rs_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

    # Saving random forest model
    try:
        joblib.dump(best_rf, "best_rf_model.pkl")
        print("Saved RandomForest pipeline to 'best_rf_model.pkl'")
    except Exception as e:
        print("Failed to save RandomForest model:", e)

    # XGBoost model
    xgb_available = False
    best_xgb = None
    y_pred_xgb = None
    y_proba_xgb = None
    try:
        from xgboost import XGBClassifier
        print("\nXGBoost detected. Training XGBoost (RandomizedSearchCV)...")
        pipe_xgb = Pipeline(steps=[("pre", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state))])
        param_dist_xgb = {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 6, 8], "clf__learning_rate": [0.01, 0.1, 0.2]}
        rs_xgb = RandomizedSearchCV(pipe_xgb, param_dist_xgb, n_iter=6, cv=3, scoring="roc_auc", random_state=random_state, n_jobs=-1)
        rs_xgb.fit(X_train, y_train)
        best_xgb = rs_xgb.best_estimator_
        y_pred_xgb = best_xgb.predict(X_test)
        y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

        # Save XGBoost model
        try:
            joblib.dump(best_xgb, "best_xgb_model.pkl")
            print("Saved XGBoost pipeline to 'best_xgb_model.pkl'")
        except Exception as e:
            print("Failed to save XGBoost model:", e)

        xgb_available = True
    except Exception:
        print("\nXGBoost not available, skipping XGBoost.")

    def eval_metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        }

    results = {"LogisticRegression": eval_metrics(y_test, y_pred_lr, y_proba_lr), "RandomForest": eval_metrics(y_test, y_pred_rf, y_proba_rf)}
    if xgb_available:
        results["XGBoost"] = eval_metrics(y_test, y_pred_xgb, y_proba_xgb)

    results_df = pd.DataFrame(results).T
    results_df = results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    print("\nModel performance on TEST set:")
    print(results_df.round(4))

    # Plot ROC curves for all available models
    plt.figure(figsize=(8, 6))
    lw = 2

    # Logistic ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.plot(fpr_lr, tpr_lr, lw=lw, label=f"LogisticRegression (AUC = {roc_auc_lr:.4f})")

    # RandomForest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, lw=lw, label=f"RandomForest (AUC = {roc_auc_rf:.4f})")

    # XGBoost ROC (if available)
    if xgb_available and y_proba_xgb is not None:
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
        plt.plot(fpr_xgb, tpr_xgb, lw=lw, label=f"XGBoost (AUC = {roc_auc_xgb:.4f})")

    # Plot formatting
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_plot_path = "roc_curves.png"
    plt.tight_layout()
    plt.savefig(roc_plot_path)
    try:
        plt.show()
    except Exception:
        pass
    plt.clf()
    print(f"Saved ROC plot to: {roc_plot_path}")

    # Feature importances from RandomForest
    try:
        ohe = rs_rf.best_estimator_.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
        try:
            ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))
        except Exception:
            ohe_feature_names = []
    except Exception:
        ohe_feature_names = []

    all_feature_names = numeric_features + ohe_feature_names
    try:
        rf_clf = rs_rf.best_estimator_.named_steps["clf"]
        importances = rf_clf.feature_importances_
        fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
        print("\nTop 20 feature importances (RandomForest):")
        print(fi.head(20).to_string())
    except Exception:
        print("\nFeature importance extraction failed.")

    if data_out_path:
        try:
            df.to_csv(data_out_path, index=False)
            print(f"\nSaved engineered dataset to: {data_out_path}")
        except Exception as e:
            print(f"\nFailed to save engineered dataset: {e}")

    try:
        print("\nRandomForest best params:", rs_rf.best_params_)
    except Exception:
        pass

    if xgb_available:
        try:
            print("XGBoost best params:", rs_xgb.best_params_)
        except Exception:
            pass

    # final classification report for best model by ROC-AUC
    try:
        best_name = results_df["roc_auc"].idxmax()
        print("\nBest model by ROC-AUC:", best_name)
        if best_name == "RandomForest":
            print(classification_report(y_test, y_pred_rf, digits=4))
        elif best_name == "LogisticRegression":
            print(classification_report(y_test, y_pred_lr, digits=4))
        elif best_name == "XGBoost" and xgb_available:
            print(classification_report(y_test, y_pred_xgb, digits=4))
    except Exception:
        pass

    return {"results_df": results_df, "rs_rf": rs_rf, "best_estimators": {"lr": pipe_lr, "rf": best_rf, "xgb": (best_xgb if 'best_xgb' in locals() else None)}}


def main():
    DATA_PATH = "online_shoppers_intention.csv"
    OUT_PATH = "online_shoppers_cleaned_featured.csv"
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)
    print("Original shape:", df.shape)
    print("First 5 rows:")
    print(df.head().to_string())

    print("\nPreparing and engineering features...")
    df = prepare_and_engineer(df)
    print("Shape after engineering:", df.shape)

    print("\nRunning correlation analysis (numeric features)...")
    correlation_analysis(df, save_path="correlation_heatmap_full.png", show_plot=True)

    print("\nRunning build and evaluate...")
    build_and_evaluate(df, data_out_path=OUT_PATH)


if __name__ == "__main__":
    main()