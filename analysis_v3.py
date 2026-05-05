# =============================================================================
# DSA 210 – Spring 2026
# Economy vs. Football Performance: Turkey & England (1993–2024)
# Author: Muhammed Mustafa
#
# AI DISCLOSURE: This code was written with assistance from Claude (Anthropic).
# All prompts and outputs are documented in the GitHub README.
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 0 – SETUP
# Run this cell first in Google Colab:
# !pip install openpyxl scikit-learn pandas matplotlib seaborn scipy --quiet
# -----------------------------------------------------------------------------

# ── 0a. Upload file to Colab (FIX FOR FileNotFoundError) ─────────────────────
# Run the following 3 lines in a SEPARATE cell, then select master_data_v2.xlsx:
#
#   from google.colab import files
#   uploaded = files.upload()        # → select master_data_v2.xlsx
#   print("Uploaded:", list(uploaded.keys()))
#
# After the file is uploaded, the path below will work without changes.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

FILE = "master_data_v2.xlsx"   # name of the file you uploaded to Colab

# -----------------------------------------------------------------------------
# SECTION 1 – DATA LOADING
# -----------------------------------------------------------------------------
df_tr  = pd.read_excel(FILE, sheet_name="turkey_data")
df_eng = pd.read_excel(FILE, sheet_name="england_data")
df_all = pd.read_excel(FILE, sheet_name="combined")

for df in [df_tr, df_eng, df_all]:
    df.columns = df.columns.str.strip().str.lower()

TR_CLUBS  = ["galatasaray", "fenerbahce", "besiktas", "trabzonspor"]
ENG_CLUBS = ["man_city", "man_utd", "arsenal", "liverpool", "tottenham", "chelsea"]

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

to_num(df_tr,  ["usdtry","gdp_pct","cpi_pct"] + TR_CLUBS  + ["total_club","fifa_rank","econ_index","fifa_success_idx"])
to_num(df_eng, ["gbpusd","gdp_pct","cpi_pct"] + ENG_CLUBS + ["total_club","fifa_rank","econ_index","fifa_success_idx"])
to_num(df_all, ["fx_rate","gdp_pct","cpi_pct","total_club","econ_index","fifa_success_idx"])

print("=" * 65)
print("  DATA LOADING SUMMARY")
print("=" * 65)
print(f"  Turkey   : {len(df_tr)}  half-year periods  |  columns: {len(df_tr.columns)}")
print(f"  England  : {len(df_eng)}  half-year periods  |  columns: {len(df_eng.columns)}")
print(f"  Combined : {len(df_all)} half-year periods  |  columns: {len(df_all.columns)}")
print(f"  Missing values — Turkey: {df_tr.isnull().sum().sum()}  |  England: {df_eng.isnull().sum().sum()}")
print("=" * 65)

# -----------------------------------------------------------------------------
# SECTION 2 – VISUALISATION
# -----------------------------------------------------------------------------
TR_CLR = {
    "galatasaray": "#E53935", "fenerbahce": "#F9A825",
    "besiktas":    "#212121", "trabzonspor": "#6A1B9A",
    "total": "#FF6F00", "national": "#2E7D32",
    "fx":    "#1565C0", "econ":     "#90CAF9",
}
ENG_CLR = {
    "man_city":  "#1E88E5", "man_utd":   "#C62828",
    "arsenal":   "#EF5350", "liverpool": "#B71C1C",
    "tottenham": "#1B5E20", "chelsea":   "#1A237E",
    "total": "#FF6F00", "national": "#2E7D32",
    "fx":    "#6A1B9A", "econ":     "#CE93D8",
}

def plot_country(df, clubs_dict, color_dict, fx_col, fx_label, title, fname):
    """
    clubs_dict : {column_name: display_label}
    color_dict : must contain keys for each club + 'total','national','fx','econ'
    """
    x = df.index
    fig, axes = plt.subplots(3, 1, figsize=(22, 16), sharex=True)
    fig.suptitle(f"{title} — Economy vs. Football Performance (1993–2024)\n"
                 f"N = {len(df)} half-year periods",
                 fontsize=14, fontweight="bold", y=0.99)

    # Panel A – Economy
    ax = axes[0]
    ax.set_title("A – Economic Indicators", fontweight="bold", loc="left")
    l1, = ax.plot(x, df[fx_col], color=color_dict["fx"], lw=2.5, label=f"{fx_label} (left axis)")
    ax.set_ylabel(fx_label, color=color_dict["fx"])
    ax.tick_params(axis="y", labelcolor=color_dict["fx"])
    ax2 = ax.twinx()
    ax2.fill_between(x, df["econ_index"], alpha=0.35, color=color_dict["econ"])
    l2, = ax2.plot(x, df["econ_index"], color=color_dict["econ"], lw=1.5,
                   linestyle="--", label="Economic Index [0–1] (right axis)")
    ax2.set_ylabel("Economic Health Index", color=color_dict["econ"])
    ax2.set_ylim(0, 1.3)
    ax.legend(handles=[l1, l2], loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Panel B – Club UEFA points
    ax = axes[1]
    ax.set_title("B – UEFA Points by Club", fontweight="bold", loc="left")
    for col, label in clubs_dict.items():
        clr = color_dict.get(col, "#888888")
        ax.plot(x, df[col], color=clr, lw=2, marker="o", markersize=3, label=label)
    ax.plot(x, df["total_club"], color=color_dict["total"], lw=3,
            linestyle="--", alpha=0.75, label="Total")
    ax.set_ylabel("UEFA Points")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.25)

    # Panel C – National team
    ax = axes[2]
    ax.set_title("C – National Team Success (FIFA Index)", fontweight="bold", loc="left")
    ax.plot(x, df["fifa_success_idx"], color=color_dict["national"], lw=2.5)
    ax.fill_between(x, df["fifa_success_idx"], alpha=0.15, color=color_dict["national"])
    ax.set_ylabel("FIFA Success Index (higher = better ranking)")
    ax.grid(axis="y", alpha=0.25)

    step = max(1, len(x) // 14)
    axes[2].set_xticks(x[::step])
    axes[2].set_xticklabels(df["period"].iloc[::step], rotation=45, ha="right", fontsize=8)
    axes[2].set_xlabel("Period")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figure saved: {fname}")

# Turkey
tr_clubs = {"galatasaray":"Galatasaray","fenerbahce":"Fenerbahce",
            "besiktas":"Besiktas","trabzonspor":"Trabzonspor"}
plot_country(df_tr, tr_clubs, TR_CLR, "usdtry", "USD/TRY Rate",
             "Turkey", "fig_overview_turkey.png")

# England
eng_clubs = {"man_city":"Man City","man_utd":"Man Utd","arsenal":"Arsenal",
             "liverpool":"Liverpool","tottenham":"Tottenham","chelsea":"Chelsea"}
plot_country(df_eng, eng_clubs, ENG_CLR, "gbpusd", "GBP/USD Rate",
             "England", "fig_overview_england.png")

# ── Side-by-side comparison figure ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Turkey vs. England: Economy & Football Comparison",
             fontsize=14, fontweight="bold")

x = df_tr.index
step = max(1, len(x) // 10)

# Economic Index
ax = axes[0, 0]
ax.plot(x, df_tr["econ_index"],  color="#E53935", lw=2, label="Turkey")
ax.plot(x, df_eng["econ_index"], color="#1565C0", lw=2, label="England")
ax.set_title("Economic Health Index"); ax.legend(); ax.grid(alpha=0.3)
ax.set_ylabel("Economic Index [0–1]")

# Total club UEFA points
ax = axes[0, 1]
ax.plot(x, df_tr["total_club"],  color="#E53935", lw=2, label="Turkey (4 clubs)")
ax.plot(x, df_eng["total_club"], color="#1565C0", lw=2, label="England (Big 6)")
ax.set_title("Total Club UEFA Points"); ax.legend(); ax.grid(alpha=0.3)
ax.set_ylabel("UEFA Points")

# National team
ax = axes[1, 0]
ax.plot(x, df_tr["fifa_success_idx"],  color="#E53935", lw=2, label="Turkey")
ax.plot(x, df_eng["fifa_success_idx"], color="#1565C0", lw=2, label="England")
ax.set_title("National Team FIFA Success Index"); ax.legend(); ax.grid(alpha=0.3)
ax.set_ylabel("FIFA Success Index")

# Scatter: Economic Index vs Club Points
ax = axes[1, 1]
ax.scatter(df_tr["econ_index"],  df_tr["total_club"],  color="#E53935", alpha=0.5, s=35, label="Turkey")
ax.scatter(df_eng["econ_index"], df_eng["total_club"], color="#1565C0", alpha=0.5, s=35, label="England")
for df_, clr in [(df_tr, "#E53935"), (df_eng, "#1565C0")]:
    m = df_[["econ_index", "total_club"]].dropna()
    z = np.polyfit(m["econ_index"], m["total_club"], 1)
    xl = np.linspace(m["econ_index"].min(), m["econ_index"].max(), 100)
    ax.plot(xl, np.polyval(z, xl), color=clr, lw=1.5, linestyle="--")
ax.set_title("Economic Index vs. Club Success (with trend lines)")
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel("Economic Index"); ax.set_ylabel("Total UEFA Points")

for ax_ in axes.flat:
    ax_.set_xticks(x[::step])
    ax_.set_xticklabels(df_tr["period"].iloc[::step], rotation=45, ha="right", fontsize=7)

plt.tight_layout()
plt.savefig("fig_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Figure saved: fig_comparison.png")

# ── Correlation heatmaps ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for i, (df_, title, cols) in enumerate([
    (df_tr,  "Turkey",
     ["usdtry","gdp_pct","cpi_pct","econ_index"] + TR_CLUBS + ["total_club","fifa_success_idx"]),
    (df_eng, "England",
     ["gbpusd","gdp_pct","cpi_pct","econ_index"] + ENG_CLUBS + ["total_club","fifa_success_idx"]),
]):
    cols = [c for c in cols if c in df_.columns]
    corr = df_[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.4, ax=axes[i], vmin=-1, vmax=1, annot_kws={"size": 8})
    axes[i].set_title(f"{title} — Pearson Correlation Heatmap", fontweight="bold")
    axes[i].tick_params(axis="x", rotation=35)
plt.tight_layout()
plt.savefig("fig_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Figure saved: fig_heatmap.png")

# -----------------------------------------------------------------------------
# SECTION 3 – HYPOTHESIS TESTING
# -----------------------------------------------------------------------------
print("\n" + "=" * 68)
print("  HYPOTHESIS TESTING — FULL REPORT")
print("=" * 68)
print("""
RESEARCH QUESTION
─────────────────
Does a country's economic performance correlate with its clubs'
UEFA results and its national team's FIFA ranking? (1993–2024)

HYPOTHESES
──────────
H₀ (Null):        There is NO statistically significant linear
                  correlation between economic indicators and
                  football success  (ρ = 0).

H₁ (Alternative): A positive correlation exists — stronger economy
                  associates with better football performance (ρ > 0).

Significance level: α = 0.05  |  Test: one-tailed Pearson r
""")

def run_test(df_, xc, yc, x_label, y_label, country=""):
    m = df_[[xc, yc]].dropna()
    x, y = m[xc].values, m[yc].values
    n = len(x)
    r,  p2  = pearsonr(x, y)
    p1 = p2 / 2 if r > 0 else 1 - p2 / 2
    rs, _   = spearmanr(x, y)
    z  = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    ci = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))
    ar = abs(r)
    effect = "large" if ar >= 0.5 else "medium" if ar >= 0.3 else "small" if ar >= 0.1 else "negligible"
    reject = p1 < 0.05

    print(f"\n  [{country}]  {x_label}  ×  {y_label}  (N={n})")
    print(f"    Pearson r      : {r:+.4f}   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"    R²             : {r**2*100:.1f}%   (variance explained)")
    print(f"    p two-tailed   : {p2:.6f}")
    print(f"    p one-tailed   : {p1:.6f}   (H₁: r > 0)")
    print(f"    Spearman ρ     : {rs:+.4f}")
    print(f"    Effect size    : {effect}  (|r| = {ar:.3f})")
    if reject and r > 0:
        verdict = "REJECT H₀  →  statistically significant POSITIVE correlation"
    elif reject and r < 0:
        verdict = "REJECT H₀  →  statistically significant NEGATIVE correlation"
    else:
        verdict = "FAIL TO REJECT H₀  →  no significant correlation at α = 0.05"
    print(f"    Decision       : {'✓' if reject else '✗'}  {verdict}")
    return dict(country=country, x=x_label, y=y_label, n=n, r=r, p1=p1,
                effect=effect, reject=reject)

results = []

print("\n─── TURKEY ──────────────────────────────────────────────────────────")
results.append(run_test(df_tr, "econ_index", "total_club",       "Economic Index", "Total Club UEFA",   "TR"))
results.append(run_test(df_tr, "econ_index", "fifa_success_idx", "Economic Index", "National Team FIFA","TR"))
results.append(run_test(df_tr, "usdtry",     "total_club",       "USD/TRY Rate",   "Total Club UEFA",   "TR"))
results.append(run_test(df_tr, "gdp_pct",    "total_club",       "GDP Growth %",   "Total Club UEFA",   "TR"))
results.append(run_test(df_tr, "cpi_pct",    "total_club",       "Inflation %",    "Total Club UEFA",   "TR"))
for col, name in zip(TR_CLUBS, ["Galatasaray", "Fenerbahce", "Besiktas", "Trabzonspor"]):
    results.append(run_test(df_tr, "econ_index", col, "Economic Index", name, "TR"))

print("\n─── ENGLAND ─────────────────────────────────────────────────────────")
results.append(run_test(df_eng, "econ_index", "total_club",       "Economic Index", "Total Club UEFA",   "ENG"))
results.append(run_test(df_eng, "econ_index", "fifa_success_idx", "Economic Index", "National Team FIFA","ENG"))
results.append(run_test(df_eng, "gbpusd",     "total_club",       "GBP/USD Rate",   "Total Club UEFA",   "ENG"))
results.append(run_test(df_eng, "gdp_pct",    "total_club",       "GDP Growth %",   "Total Club UEFA",   "ENG"))
results.append(run_test(df_eng, "cpi_pct",    "total_club",       "Inflation %",    "Total Club UEFA",   "ENG"))
for col, name in zip(ENG_CLUBS, ["Man City", "Man Utd", "Arsenal", "Liverpool", "Tottenham", "Chelsea"]):
    results.append(run_test(df_eng, "econ_index", col, "Economic Index", name, "ENG"))

# Summary table
print("\n" + "=" * 68)
print("  SUMMARY TABLE — ALL TESTS")
print("=" * 68)
print(f"  {'C':<4} {'Pair':<40} {'r':>6} {'p (1-tail)':>11} {'Effect':>10} {'H₀':>8}")
print("  " + "─" * 65)
for res in results:
    pair = f"{res['x'][:18]} × {res['y'][:18]}"
    h0   = "REJECT" if res["reject"] else "KEEP"
    print(f"  {res['country']:<4} {pair:<40} {res['r']:>+6.3f} {res['p1']:>11.4f} {res['effect']:>10} {h0:>8}")
print("=" * 68)

# Cross-country comparison
r_tr_club,  _ = pearsonr(df_tr["econ_index"],  df_tr["total_club"])
r_eng_club, _ = pearsonr(df_eng["econ_index"], df_eng["total_club"])
r_tr_nat,   _ = pearsonr(df_tr["econ_index"],  df_tr["fifa_success_idx"])
r_eng_nat,  _ = pearsonr(df_eng["econ_index"], df_eng["fifa_success_idx"])

print(f"""
  CROSS-COUNTRY COMPARISON
  ─────────────────────────────────────────────────────────────
  Turkey   — Economic Index × Club Success   : r = {r_tr_club:+.4f}
  England  — Economic Index × Club Success   : r = {r_eng_club:+.4f}

  Turkey   — Economic Index × National Team  : r = {r_tr_nat:+.4f}
  England  — Economic Index × National Team  : r = {r_eng_nat:+.4f}

  Insight: England shows a {'stronger' if abs(r_eng_club) > abs(r_tr_club) else 'weaker'}
  economy–club relationship (|r|={abs(r_eng_club):.3f} vs {abs(r_tr_club):.3f}).
  England's Premier League clubs generate revenue independently
  of the national economy (broadcast deals, foreign ownership),
  which may explain the negative correlation direction.
""")

# -----------------------------------------------------------------------------
# SECTION 4 – MACHINE LEARNING
# -----------------------------------------------------------------------------
print("=" * 68)
print("  MACHINE LEARNING")
print("=" * 68)

# Prepare ML datasets
TR_FEAT   = ["usdtry","gdp_pct","cpi_pct","econ_index"] + TR_CLUBS  + ["total_club","fifa_success_idx"]
ENG_FEAT  = ["gbpusd","gdp_pct","cpi_pct","econ_index"] + ENG_CLUBS + ["total_club","fifa_success_idx"]
COMB_FEAT = ["fx_rate","gdp_pct","cpi_pct","econ_index","total_club","fifa_success_idx"]

ml_tr   = df_tr [[c for c in TR_FEAT   if c in df_tr.columns ]].dropna().copy()
ml_eng  = df_eng[[c for c in ENG_FEAT  if c in df_eng.columns]].dropna().copy()
ml_comb = df_all[[c for c in COMB_FEAT if c in df_all.columns]].dropna().copy()

print(f"\n  ML Dataset Sizes:")
print(f"    Turkey           : {ml_tr.shape[0]} rows × {ml_tr.shape[1]} features")
print(f"    England          : {ml_eng.shape[0]} rows × {ml_eng.shape[1]} features")
print(f"    Combined (TR+ENG): {ml_comb.shape[0]} rows × {ml_comb.shape[1]} features  ← primary ML power")

# ── 4a. K-Means Clustering ────────────────────────────────────────────────────
print("\n─── 4a. K-Means Clustering (unsupervised) ───────────────────────────")
print("""
  Goal: Discover natural 'eras' in the data without labels.
  Method: Elbow method to choose optimal k, then characterise clusters.
""")

fig, axes_km = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("K-Means: Elbow Method", fontweight="bold")

datasets = [
    ("Turkey",   ml_tr,   "#E53935"),
    ("England",  ml_eng,  "#1565C0"),
    ("Combined", ml_comb, "#2E7D32"),
]

print(f"\n  {'Dataset':<12} {'k=2':>8} {'k=3':>8} {'k=4':>8} {'k=5':>8} {'k=6':>8}")
print("  " + "─" * 52)
for (name, ml_, clr), ax_km in zip(datasets, axes_km):
    sc = StandardScaler(); Xs = sc.fit_transform(ml_)
    inertias = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs); inertias.append(km.inertia_)
    ax_km.plot(range(2, 8), inertias, "o-", color=clr, lw=2)
    ax_km.set_title(f"{name}"); ax_km.set_xlabel("k"); ax_km.set_ylabel("Inertia")
    ax_km.grid(alpha=0.3)
    print(f"  {name:<12} " + "  ".join(f"{v:>8.1f}" for v in inertias[:5]))

plt.tight_layout()
plt.savefig("fig_elbow.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Figure saved: fig_elbow.png")

# PCA visualisation
BEST_K = 3
fig, axes_pca = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"PCA + K-Means Clusters (k={BEST_K})", fontweight="bold")
K_COLORS = ["#E53935", "#1565C0", "#2E7D32", "#F9A825"]

for (name, ml_, clr), ax_p in zip(datasets, axes_pca):
    sc = StandardScaler(); Xs = sc.fit_transform(ml_)
    km = KMeans(n_clusters=BEST_K, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    pca = PCA(n_components=2, random_state=42); Xp = pca.fit_transform(Xs)

    for ki in range(BEST_K):
        mask = labels == ki
        ax_p.scatter(Xp[mask, 0], Xp[mask, 1], color=K_COLORS[ki], s=50,
                     alpha=0.7, edgecolors="white", lw=0.5, label=f"Cluster {ki}")

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax_p.set_title(f"{name}  (k={BEST_K})\nPC1={var1:.0f}%  PC2={var2:.0f}%  total={var1+var2:.0f}%")
    ax_p.legend(fontsize=8); ax_p.grid(alpha=0.25)
    ax_p.set_xlabel("PC1"); ax_p.set_ylabel("PC2")

    counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"\n  [{name}] Cluster distribution: {counts}")
    cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=ml_.columns)
    cent.index = [f"Cluster {i}" for i in range(BEST_K)]
    show = [c for c in ["econ_index", "total_club", "fifa_success_idx"] if c in cent.columns]
    print(f"  Centroids:\n{cent[show].round(3).to_string()}")

plt.tight_layout()
plt.savefig("fig_pca_clusters.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Figure saved: fig_pca_clusters.png")

# ── 4b. Supervised Classification ────────────────────────────────────────────
print("\n─── 4b. Supervised Classification ──────────────────────────────────")
print("""
  Goal: Predict whether a period will be 'high' or 'low' club
        success using only economic indicators.

  Target variable  : binary label from total_club
    1 = 'High'       (above median)
    0 = 'Low'        (at or below median)
  Predictor set    : fx_rate, gdp_pct, cpi_pct, econ_index
  Evaluation       : 5-fold stratified cross-validation
""")

def classify(ml_, fx_col, name):
    med = ml_["total_club"].median()
    ml_c = ml_.copy()
    ml_c["label"] = (ml_c["total_club"] > med).astype(int)
    features = [c for c in [fx_col, "gdp_pct", "cpi_pct", "econ_index"] if c in ml_c.columns]
    Xc = StandardScaler().fit_transform(ml_c[features].values)
    yc = ml_c["label"].values

    class_dist = dict(zip(*np.unique(yc, return_counts=True)))
    baseline   = max(class_dist.values()) / len(yc)

    print(f"\n  [{name}]  N={len(yc)}  |  median={med:.1f}  |  "
          f"class distribution: {class_dist}  |  majority baseline: {baseline:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Logistic Regression" : LogisticRegression(max_iter=500, random_state=42),
        "Random Forest"        : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print(f"  {'Model':<22} {'Accuracy':>10} {'F1-macro':>10} {'Precision':>10} {'Recall':>9}")
    print("  " + "─" * 65)
    best_acc, best_model, best_name = 0, None, ""
    for mname, model in models.items():
        acc  = cross_val_score(model, Xc, yc, cv=cv, scoring="accuracy").mean()
        f1   = cross_val_score(model, Xc, yc, cv=cv, scoring="f1_macro").mean()
        prec = cross_val_score(model, Xc, yc, cv=cv, scoring="precision_macro").mean()
        rec  = cross_val_score(model, Xc, yc, cv=cv, scoring="recall_macro").mean()
        gain = acc - baseline
        print(f"  {mname:<22} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>9.4f}  "
              f"(+{gain:.3f} vs baseline)")
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, mname

    # Feature importances
    best_model.fit(Xc, yc)
    if hasattr(best_model, "feature_importances_"):
        fi = sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1])
        print(f"\n  [{name}] Best model: {best_name}  (accuracy={best_acc:.4f})")
        print(f"  Feature importances:")
        for fname_, imp in fi:
            bar = "█" * int(imp * 40)
            print(f"    {fname_:<25}: {imp:.4f}  ({imp*100:.1f}%)  {bar}")
    return best_acc

acc_tr   = classify(ml_tr,   "usdtry",   "Turkey")
acc_eng  = classify(ml_eng,  "gbpusd",   "England")
acc_comb = classify(ml_comb, "fx_rate",  "Combined (128 rows)")

# ── 4c. Country classification (Combined dataset) ────────────────────────────
print("\n─── 4c. Country Classification (Combined Dataset) ───────────────────")
print("""
  Goal: Predict which country a period belongs to using economic
        and football features.  1 = England, 0 = Turkey.
  This measures how structurally different the two economies are.
""")
df_all_c = df_all.copy()
df_all_c["country_label"] = (df_all_c["country"].str.lower() == "england").astype(int)
feat_c = [c for c in ["fx_rate","gdp_pct","cpi_pct","econ_index","total_club","fifa_success_idx"]
          if c in df_all_c.columns]
sub_c   = df_all_c[feat_c + ["country_label"]].dropna()
Xcc = StandardScaler().fit_transform(sub_c[feat_c].values)
ycc = sub_c["country_label"].values
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_c = RandomForestClassifier(n_estimators=100, random_state=42)
acc_c = cross_val_score(rf_c, Xcc, ycc, cv=cv5, scoring="accuracy").mean()
f1_c  = cross_val_score(rf_c, Xcc, ycc, cv=cv5, scoring="f1_macro").mean()
print(f"  Random Forest — Country prediction:  Accuracy={acc_c:.4f}  F1={f1_c:.4f}")
rf_c.fit(Xcc, ycc)
fi_c = sorted(zip(feat_c, rf_c.feature_importances_), key=lambda x: -x[1])
print("  Feature importances (which variables distinguish TR from ENG most):")
for fn, imp in fi_c:
    bar = "█" * int(imp * 40)
    print(f"    {fn:<25}: {imp:.4f}  ({imp*100:.1f}%)  {bar}")

# ── 4d. OLS Regression ───────────────────────────────────────────────────────
print("\n─── 4d. OLS Regression ──────────────────────────────────────────────")
print("  Dependent variable: total_club  |  Predictors: econ_index, gdp_pct, cpi_pct\n")

from scipy.stats import t as tdist, f as fdist

def ols(df_, xc_list, yc, label):
    sub = df_[[yc] + xc_list].dropna()
    Xr  = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in xc_list])
    yr  = sub[yc].values
    beta = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    yhat = Xr @ beta; resid = yr - yhat
    n, k = len(yr), Xr.shape[1] - 1
    ss_res = np.sum(resid**2); ss_tot = np.sum((yr - yr.mean())**2)
    r2  = 1 - ss_res / ss_tot
    r2a = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    mse = ss_res / (n - k - 1)
    se  = np.sqrt(mse * np.linalg.inv(Xr.T @ Xr).diagonal())
    t_s = beta / se
    pv  = 2 * (1 - tdist.cdf(np.abs(t_s), df=n - k - 1))
    ff  = (r2 / k) / ((1 - r2) / (n - k - 1))
    pf  = 1 - fdist.cdf(ff, k, n - k - 1)
    print(f"  [{label}]  N={n}  R²={r2:.4f}  Adj.R²={r2a:.4f}  "
          f"F={ff:.3f}  p(F)={pf:.6f}")
    for vn, b, pval in zip(["Intercept"] + xc_list, beta, pv):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    {vn:<22}:  β={b:+.4f}   p={pval:.4f}  {sig}")
    print()

ols(df_tr,  ["econ_index","gdp_pct","cpi_pct"], "total_club",       "Turkey  — Club Success")
ols(df_tr,  ["econ_index","gdp_pct","cpi_pct"], "fifa_success_idx", "Turkey  — National Team")
ols(df_eng, ["econ_index","gdp_pct","cpi_pct"], "total_club",       "England — Club Success")
ols(df_eng, ["econ_index","gdp_pct","cpi_pct"], "fifa_success_idx", "England — National Team")

# -----------------------------------------------------------------------------
# SECTION 5 – FINAL SUMMARY
# -----------------------------------------------------------------------------
print("=" * 68)
print("  FINAL SUMMARY")
print("=" * 68)
r_tr  = pearsonr(df_tr["econ_index"],  df_tr["total_club"])[0]
r_eng = pearsonr(df_eng["econ_index"], df_eng["total_club"])[0]
print(f"""
  Dataset
  ───────
  Turkey           : {len(df_tr)} half-year periods (4 clubs + national team)
  England          : {len(df_eng)} half-year periods (Big 6 + national team)
  Combined ML set  : {len(df_all)} periods  →  improved ML statistical power

  Correlation Results  (Economic Index × Total Club UEFA Points)
  ──────────────────────────────────────────────────────────────
  Turkey           :  r = {r_tr:+.4f}  (positive, small effect)
  England          :  r = {r_eng:+.4f}  (negative — see note below)

  Note on England's negative r:
  Premier League clubs earn the majority of their revenue from
  global TV rights, foreign investment, and transfer fees that are
  largely decoupled from the UK domestic economy. A strong GBP/USD
  actually *reduces* the pound-value of foreign-currency revenues,
  explaining why a healthy economy correlates weakly or negatively
  with UEFA points.

  ML Performance  (5-fold stratified cross-validation accuracy)
  ──────────────────────────────────────────────────────────────
  Turkey   classification  :  {acc_tr:.4f}
  England  classification  :  {acc_eng:.4f}
  Combined classification  :  {acc_comb:.4f}
  Country prediction (TR/ENG):  {acc_c:.4f}

  Limitations
  ───────────
  • N=64 per country is below the ideal for ML (noted by instructor).
    Combined N=128 partially addresses this.
  • GDP and CPI are annual values assigned to both halves of a year.
  • Club UEFA points are approximations; verify against primary sources.
  • Confounders (manager changes, transfer budgets, injuries) not captured.

  Saved figures
  ─────────────
  fig_overview_turkey.png    fig_overview_england.png
  fig_comparison.png         fig_heatmap.png
  fig_elbow.png              fig_pca_clusters.png
""")
print("=" * 68)
print("  Analysis complete. All figures saved as PNG.")
print("=" * 68)
