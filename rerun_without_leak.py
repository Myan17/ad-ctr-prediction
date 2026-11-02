"""Re-run the notebook's models with and without the leaked feature `item_ctr_rank`.

Reproduces ctr_analysis.ipynb exactly (same 500,000 rows, aggregation, features, split, models, search space,
and seeds), then repeats every model with `item_ctr_rank` removed. A predict-the-training-mean / majority-class
baseline is added for scale.

Usage: python rerun_without_leak.py path/to/train.gz
Writes results/rerun_results.md and images/feature_importance_no_leak.png.
"""
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_SPACE = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
ALL_FEATURES = ["log_pv", "banner_pos", "is_app", "device_type", "conn_type",
                "dow", "is_weekend", "log_creatives_per_item", "item_ctr_rank"]


def load(path):
    str_cols = ["id", "site_id", "site_domain", "site_category", "app_id", "app_domain",
                "app_category", "device_id", "device_ip", "device_model"]
    raw = pd.read_csv(path, nrows=500_000, dtype={c: str for c in str_cols})
    raw["ds"] = pd.to_datetime(raw["hour"].astype(str), format="%y%m%d%H")
    raw["date"] = raw["ds"].dt.normalize()
    raw["is_app"] = (raw["site_id"] == "85f751fd").astype(int)
    raw["item_id"] = np.where(raw["is_app"] == 1, raw["app_id"], raw["site_id"])
    raw["creative_key"] = raw["item_id"] + "_" + raw["banner_pos"].astype(str) + "_" + raw["C1"].astype(str)
    raw["creative_id"] = LabelEncoder().fit_transform(raw["creative_key"])
    raw["item_id_enc"] = LabelEncoder().fit_transform(raw["item_id"])

    meta = raw.groupby("creative_id").agg(
        item_id=("item_id_enc", "first"),
        banner_pos=("banner_pos", "first"),
        is_app=("is_app", "first"),
        device_type=("device_type", lambda x: x.mode().iloc[0]),
        conn_type=("device_conn_type", lambda x: x.mode().iloc[0]),
    ).reset_index()
    daily = raw.groupby(["creative_id", "date"]).agg(pv=("click", "count"), clk=("click", "sum")).reset_index()
    df = daily.merge(meta, on="creative_id")
    df["ds"] = pd.to_datetime(df["date"])
    df = df[df["pv"] >= 10].copy().reset_index(drop=True)
    df["CTR"] = df["clk"] / df["pv"]

    df["log_pv"] = np.log1p(df["pv"])
    df["dow"] = df["ds"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["log_creatives_per_item"] = np.log1p(df.groupby("item_id")["creative_id"].transform("count"))
    df["item_ctr_rank"] = df.groupby("item_id")["CTR"].transform("median").rank(pct=True)
    df["high_ctr"] = (df["CTR"] > df["CTR"].median()).astype(int)
    return df


def reg_metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5, "R2": r2_score(y, p)}


def clf_metrics(y, pred, proba):
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "ROC-AUC": roc_auc_score(y, proba)}


def run(df, features):
    X = df[features]
    Xtr, Xte, ytr_r, yte_r = train_test_split(X, df["CTR"], test_size=0.2, random_state=42)
    _, _, ytr_c, yte_c = train_test_split(X, df["high_ctr"], test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    reg = {"Baseline (training mean)": reg_metrics(yte_r, np.full(len(yte_r), ytr_r.mean()))}
    reg["Linear Regression"] = reg_metrics(yte_r, LinearRegression().fit(Xtr_s, ytr_r).predict(Xte_s))
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1).fit(Xtr, ytr_r)
    reg["Random Forest"] = reg_metrics(yte_r, rf.predict(Xte))
    xr = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0).fit(Xtr, ytr_r)
    reg["XGBoost"] = reg_metrics(yte_r, xr.predict(Xte))
    rs_r = RandomizedSearchCV(xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1), SEARCH_SPACE,
                              n_iter=20, scoring="neg_root_mean_squared_error", cv=3, random_state=42,
                              n_jobs=-1).fit(Xtr, ytr_r)
    best_r = rs_r.best_estimator_
    reg["XGBoost (tuned)"] = reg_metrics(yte_r, best_r.predict(Xte))

    majority = int(ytr_c.mean() >= 0.5)
    clf = {"Baseline (majority class)": clf_metrics(yte_c, np.full(len(yte_c), majority),
                                                    np.full(len(yte_c), 0.5))}
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42).fit(Xtr_s, ytr_c)
    clf["Logistic Regression"] = clf_metrics(yte_c, lr.predict(Xte_s), lr.predict_proba(Xte_s)[:, 1])
    rfc = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42,
                                 n_jobs=-1).fit(Xtr, ytr_c)
    clf["Random Forest"] = clf_metrics(yte_c, rfc.predict(Xte), rfc.predict_proba(Xte)[:, 1])
    xc = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8,
                           colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1,
                           verbosity=0).fit(Xtr, ytr_c)
    clf["XGBoost"] = clf_metrics(yte_c, xc.predict(Xte), xc.predict_proba(Xte)[:, 1])
    rs_c = RandomizedSearchCV(xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
                              SEARCH_SPACE, n_iter=20, scoring="roc_auc", cv=3, random_state=42,
                              n_jobs=-1).fit(Xtr, ytr_c)
    best_c = rs_c.best_estimator_
    clf["XGBoost (tuned)"] = clf_metrics(yte_c, best_c.predict(Xte), best_c.predict_proba(Xte)[:, 1])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = {
        "Regression RMSE": -cross_val_score(best_r, X, df["CTR"], cv=kf, scoring="neg_root_mean_squared_error"),
        "Regression R2": cross_val_score(best_r, X, df["CTR"], cv=kf, scoring="r2"),
        "Classification ROC-AUC": cross_val_score(best_c, X, df["high_ctr"], cv=kf, scoring="roc_auc"),
        "Classification F1": cross_val_score(best_c, X, df["high_ctr"], cv=kf, scoring="f1"),
    }
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return pd.DataFrame(reg).T, pd.DataFrame(clf).T, cv, importance


def section(title, reg, clf, cv, imp):
    lines = [f"## {title}", "", "### Regression (CTR), 239-creative test set", "",
             reg.round(4).to_markdown(), "", "### Classification (CTR above median)", "",
             clf.round(4).to_markdown(), "", "### 5-fold CV, tuned XGBoost", ""]
    lines += [f"- {k}: {v.mean():.4f} +/- {v.std():.4f}" for k, v in cv.items()]
    lines += ["", "### Random forest feature importance (regression)", "", imp.round(4).to_markdown(), ""]
    return "\n".join(lines)


def main():
    df = load(sys.argv[1])
    out = [f"# Re-run results\n\n{len(df):,} creatives, dates {df['ds'].min().date()} to {df['ds'].max().date()}.\n"]
    leaked = run(df, ALL_FEATURES)
    out.append(section("Original features (includes item_ctr_rank)", *leaked))
    clean_features = [f for f in ALL_FEATURES if f != "item_ctr_rank"]
    clean = run(df, clean_features)
    out.append(section("Without item_ctr_rank", *clean))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "rerun_results.md"), "w") as f:
        f.write("\n".join(out))

    imp = clean[3].sort_values()
    plt.figure(figsize=(8, 4.5))
    plt.barh(imp.index, imp.values, color="steelblue")
    plt.xlabel("Importance (random forest, regression)")
    plt.title("Feature importance without item_ctr_rank")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "images", "feature_importance_no_leak.png"), dpi=120)
    print("\n".join(out))


if __name__ == "__main__":
    main()
