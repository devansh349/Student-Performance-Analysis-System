"""Student Performance Analysis System - Data Mining Lab (24CAP-308)

Pipeline: data generation -> audit -> cleaning -> exploration -> classification
          -> regression -> clustering -> at-risk detection -> export
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             r2_score, mean_absolute_error, silhouette_score)

SEED = 308
LEVELS = ["Low", "Medium", "High"]


# ==========================================
# 1. DATA GENERATION (synthetic student records)
# ==========================================
def generate_dataset(n=400, seed=SEED):
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(78, 12, n), 40, 100)
    study = np.clip(rng.normal(12, 5, n), 0, 35)
    prev_gpa = np.clip(rng.normal(7.0, 1.2, n), 4.0, 10.0)
    sleep = np.clip(rng.normal(6.8, 1.1, n), 3.5, 9.5)
    extra = rng.choice(["Yes", "No"], n, p=[0.45, 0.55])
    support = rng.choice(["Low", "Medium", "High"], n, p=[0.25, 0.45, 0.30])
    ability = (prev_gpa - 7.0) / 1.2
    internal = np.clip(28 + 5 * ability + 0.18 * (attendance - 78) + 0.25 * (study - 12)
                       + rng.normal(0, 4, n), 5, 50)
    s_eff = pd.Series(support).map({"Low": -3.0, "Medium": 0.0, "High": 3.0}).to_numpy()
    final = (58 + 5.5 * ability + 0.28 * (attendance - 78) + 0.75 * (study - 12)
             + 0.55 * (internal - 28) + s_eff - 3.0 * (sleep < 5.5)
             + rng.normal(0, 5.5, n))
    final = np.clip(final, 0, 100)
    df = pd.DataFrame({
        "attendance_pct": attendance.round(1), "study_hours_week": study.round(1),
        "prev_cgpa": prev_gpa.round(2), "internal_marks": internal.round(1),
        "sleep_hours": sleep.round(1), "extracurricular": extra,
        "parental_support": support, "final_score": final.round(1)})
    # inject realistic data-quality problems
    for col, k in [("study_hours_week", 20), ("sleep_hours", 16), ("internal_marks", 14)]:
        df.loc[rng.choice(n, k, replace=False), col] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "attendance_pct"] = rng.uniform(101, 130, 6).round(1)
    df.loc[rng.choice(n, 4, replace=False), "study_hours_week"] = rng.uniform(60, 90, 4).round(1)
    df.loc[rng.choice(n, 10, replace=False), "parental_support"] = np.nan
    return df


def level_of(score):
    return "Low" if score < 50 else ("Medium" if score < 70 else "High")


# ==========================================
# 2. DATA AUDIT AND CLEANING
# ==========================================
def audit(df):
    print(f"Records: {len(df)} | Attributes: {df.shape[1]}")
    miss = df.isna().sum()
    print("Missing values:", {k: int(v) for k, v in miss.items() if v})
    print(f"Attendance > 100%: {(df.attendance_pct > 100).sum()} | "
          f"Study hours > 50: {(df.study_hours_week > 50).sum()}")


def clean(df):
    df = df.copy()
    log = {}
    bad = df.attendance_pct > 100
    log["attendance_fixed"] = int(bad.sum())
    df.loc[bad, "attendance_pct"] = np.nan             # impossible value -> missing
    q1, q3 = df.study_hours_week.quantile([0.25, 0.75])
    hi = q3 + 3 * (q3 - q1)                             # extreme-outlier fence
    out = df.study_hours_week > hi
    log["study_outliers"] = int(out.sum())
    df.loc[out, "study_hours_week"] = np.nan
    log["values_imputed"] = int(df.isna().sum().sum())
    for c in ["attendance_pct", "study_hours_week", "internal_marks", "sleep_hours"]:
        df[c] = df[c].fillna(df[c].median())            # median imputation
    df["parental_support"] = df["parental_support"].fillna(df["parental_support"].mode()[0])
    df["performance_level"] = df.final_score.apply(level_of)
    return df, log


def encode(df):
    X = df.drop(columns=["final_score", "performance_level"]).copy()
    X["extracurricular"] = (X["extracurricular"] == "Yes").astype(int)
    X["parental_support"] = X["parental_support"].map({"Low": 0, "Medium": 1, "High": 2})
    return X


# ==========================================
# 3. CLASSIFICATION
# ==========================================
def classify(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "k-NN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(200, random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, m in models.items():
        cvs = cross_val_score(m, X, y, cv=cv)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        rows.append({"Model": name, "CV acc": cvs.mean(), "CV std": cvs.std(),
                     "Test acc": accuracy_score(yte, pred),
                     "Macro F1": f1_score(yte, pred, average="macro")})
    res = pd.DataFrame(rows).sort_values("CV acc", ascending=False).reset_index(drop=True)
    return res, models, (Xtr, Xte, ytr, yte)


# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    line = "=" * 78
    print(line + "\n>>> STUDENT PERFORMANCE ANALYSIS SYSTEM (DATA MINING LAB) <<<\n" + line)

    print("[1/7] Generating and auditing student dataset...")
    raw = generate_dataset()
    raw.to_csv("student_data_raw.csv", index=False)
    audit(raw)

    print("\n[2/7] Cleaning and preprocessing...")
    df, log = clean(raw)
    df.to_csv("student_data_clean.csv", index=False)
    print(f"  Impossible attendance values reset: {log['attendance_fixed']} | "
          f"Extreme study-hour outliers reset: {log['study_outliers']}")
    print(f"  Missing cells imputed (median / mode): {log['values_imputed']} | "
          f"Remaining missing: {int(df.isna().sum().sum())}")
    print("  Performance levels:", df.performance_level.value_counts().reindex(LEVELS).to_dict())

    print("\n[3/7] Exploratory analysis (correlation with final score)...")
    X = encode(df)
    corr = X.assign(final_score=df.final_score).corr()["final_score"].drop("final_score")
    for k, v in corr.sort_values(ascending=False).items():
        print(f"  {k:<18} r = {v:+.3f}")

    print("\n[4/7] Classification: predicting performance level...")
    y = df.performance_level
    res, models, (Xtr, Xte, ytr, yte) = classify(X, y)
    print(f"  {'Model':<20}{'CV acc':>8}{'CV std':>8}{'Test acc':>10}{'Macro F1':>10}")
    for _, r in res.iterrows():
        print(f"  {r['Model']:<20}{r['CV acc']:>8.3f}{r['CV std']:>8.3f}{r['Test acc']:>10.3f}{r['Macro F1']:>10.3f}")
    best = res.loc[0, "Model"]
    cm = confusion_matrix(yte, models[best].predict(Xte), labels=LEVELS)
    print(f"\n  Confusion matrix of best model ({best}); rows = actual, cols = predicted")
    print("  " + " " * 8 + "".join(f"{l:>8}" for l in LEVELS))
    for l, row in zip(LEVELS, cm):
        print(f"  {l:<8}" + "".join(f"{v:>8}" for v in row))
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X, y)
    print("\n  Decision-tree rules (depth 3):")
    for ln in export_text(tree, feature_names=list(X.columns)).splitlines():
        print("   " + ln)
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n  Random Forest feature importance:")
    for k, v in imp.items():
        print(f"   {k:<18}{v:.3f}")

    print("\n[5/7] Regression: predicting final score...")
    Xa, Xb, ya, yb = train_test_split(X, df.final_score, test_size=0.2, random_state=SEED)
    lr = LinearRegression().fit(Xa, ya)
    pr = lr.predict(Xb)
    print(f"  R2 = {r2_score(yb, pr):.3f} | MAE = {mean_absolute_error(yb, pr):.2f} marks")
    print("  Coefficients (marks per unit):")
    for c, v in zip(X.columns, lr.coef_):
        print(f"   {c:<18}{v:+.2f}")

    print("\n[6/7] Clustering students (k-means)...")
    Z = StandardScaler().fit_transform(X)
    sil = {}
    for k in range(2, 7):
        sil[k] = silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z))
    print("  Silhouette by k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in sil.items()))
    k = max(sil, key=sil.get)
    print(f"  Best k by silhouette: {k}")
    labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    df["cluster"] = labels
    prof = df.groupby("cluster")[["attendance_pct", "study_hours_week", "prev_cgpa",
                                  "internal_marks", "final_score"]].mean().round(1)
    prof.insert(0, "students", df.groupby("cluster").size())
    print(f"  Cluster profiles (k = {k}):")
    print("  " + prof.to_string().replace("\n", "\n  "))

    print("\n[7/7] At-risk detection and export...")
    cvk = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(RandomForestClassifier(200, random_state=SEED), X, y, cv=cvk, method="predict_proba")
    classes = sorted(y.unique())                       # column order of predict_proba
    proba = pd.Series(oof[:, classes.index("Low")], index=df.index)
    risk = df[(proba >= 0.5)]
    print(f"  Students flagged at-risk (P(Low) >= 0.5): {len(risk)} of {len(df)}")
    print(f"  Mean attendance of flagged: {risk.attendance_pct.mean():.1f}% "
          f"vs others: {df.drop(risk.index).attendance_pct.mean():.1f}%")
    risk.assign(p_low=proba[risk.index].round(2)).to_csv("at_risk_students.csv", index=False)
    df.to_csv("student_data_clean.csv", index=False)

    # ---------- figures ----------
    plt.figure(figsize=(6, 4.6))
    c = X.assign(final_score=df.final_score).corr()
    plt.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.xticks(range(len(c)), c.columns, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(c)), c.columns, fontsize=7)
    plt.colorbar(shrink=0.8)
    plt.title("Correlation matrix", fontsize=10)
    plt.tight_layout(); plt.savefig("fig_correlation.png", dpi=170); plt.close()

    plt.figure(figsize=(6, 4.2))
    plt.barh(res["Model"][::-1], res["CV acc"][::-1], xerr=res["CV std"][::-1], color="#1F3A8A")
    plt.xlim(0.5, 1.0); plt.xlabel("5-fold cross-validated accuracy")
    plt.title("Classifier comparison", fontsize=10)
    plt.tight_layout(); plt.savefig("fig_models.png", dpi=170); plt.close()

    p2 = PCA(2, random_state=SEED).fit_transform(Z)
    plt.figure(figsize=(6, 4.2))
    for cl, col in zip(range(k), ["#C0392B", "#1F3A8A", "#0E8F86", "#E69F00", "#7F8C8D", "#8E44AD"]):
        plt.scatter(p2[labels == cl, 0], p2[labels == cl, 1], s=14, c=col, label=f"Cluster {cl}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(fontsize=8)
    plt.title("Student clusters (PCA projection)", fontsize=10)
    plt.tight_layout(); plt.savefig("fig_clusters.png", dpi=170); plt.close()

    plt.figure(figsize=(6, 4.2))
    plt.barh(imp.index[::-1], imp.values[::-1], color="#0E8F86")
    plt.xlabel("Importance"); plt.title("Random Forest feature importance", fontsize=10)
    plt.tight_layout(); plt.savefig("fig_importance.png", dpi=170); plt.close()

    print("\n[EXPORT COMPLETE] student_data_clean.csv, at_risk_students.csv and 4 figures saved.")
    print(line)


if __name__ == "__main__":
    main()
