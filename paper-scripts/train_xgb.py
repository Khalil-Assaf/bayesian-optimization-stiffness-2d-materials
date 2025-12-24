from __future__ import annotations
import sys, json, warnings, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from xgboost import XGBRegressor
import shap

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils import seed_everything, apply_house_style

SCATTER = "#8B008B"
TREND   = "#4B0082"
IDEAL   = "#666666"
ACCENT  = "#6A5ACD"
DASH    = "#4B0082"

apply_house_style()

OUT_DIR = getattr(config, "OUT_DIR", ROOT / "outputs")
FIG_DIR = OUT_DIR / "figures"
METRICS_DIR = OUT_DIR / "metrics"
SUMMARY_DIR = OUT_DIR / "summaries"
MODELS_DIR = OUT_DIR / "models"
for d in [OUT_DIR, FIG_DIR, METRICS_DIR, SUMMARY_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "MagpieData maximum GSvolume_pa",
    "MagpieData maximum NdValence",
    "MagpieData mean MendeleevNumber",
    "MagpieData minimum Electronegativity",
    "MagpieData mean NUnfilled",
    "MagpieData mean MeltingT",
    "MagpieData mean Column",
    "MagpieData range SpaceGroupNumber",
]

SHORT = {
    "MagpieData maximum GSvolume_pa": "Max GS volume",
    "MagpieData maximum NdValence": "Max d-valence",
    "MagpieData mean MendeleevNumber": "Mean Mendeleev #",
    "MagpieData minimum Electronegativity": "Min electronegativity",
    "MagpieData mean NUnfilled": "Mean unfilled e−",
    "MagpieData mean MeltingT": "Mean melting T",
    "MagpieData mean Column": "Mean column",
    "MagpieData range SpaceGroupNumber": "Range(space-group #)",
}

def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return r2, mae, rmse

def _save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

def plot_E_histogram(y):
    y_arr = np.asarray(y).ravel()
    plt.figure(figsize=(7.0, 4.6))
    bins = np.linspace(0, float(np.ceil(y_arr.max() / 25) * 25), 50)
    plt.hist(y_arr, bins=bins, alpha=0.95, rwidth=0.9, color=ACCENT, edgecolor='black', linewidth=0.3)
    mu = float(y_arr.mean())
    p95 = float(np.percentile(y_arr, 95))
    plt.axvline(mu, ls="--", lw=1.2, color=DASH, label=f"Mean = {mu:.1f} N/m")
    plt.axvline(p95, ls=":", lw=1.2, color=DASH, label=f"95th percentile = {p95:.1f} N/m")
    plt.xlabel(r"$\bar{\mathrm{E}}$ (N/m)")
    plt.ylabel("Count")
    plt.title(r"Distribution of $\bar{\mathrm{E}}$")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "E_histogram_with_tail.png", dpi=600)
    plt.close()

def plot_feature_histograms(X: pd.DataFrame):
    rows, cols = 2, 4
    plt.figure(figsize=(12.5, 6.4))
    creative_colors = ["#8B008B","#6A5ACD","#4B0082","#7B68EE","#5D3FD3","#483D8B","#9370DB","#9932CC"]
    for i, col in enumerate(FEATURES, 1):
        ax = plt.subplot(rows, cols, i)
        x = pd.to_numeric(X[col], errors="coerce").dropna().to_numpy()
        if len(x) == 0:
            continue
        ax.hist(x, bins=30, alpha=0.95, color=creative_colors[(i - 1) % len(creative_colors)], edgecolor='black')
        ax.axvline(np.mean(x), ls="--", lw=1.0, color=DASH)
        ax.set_title(SHORT.get(col, col), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.suptitle("Feature distributions")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIG_DIR / "feats_histogram_panel.png", dpi=600)
    plt.close()

def make_feature_summary(X: pd.DataFrame, y):
    y_np = np.asarray(y).ravel()
    rows = []
    for col in FEATURES:
        x = pd.to_numeric(X[col], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y_np)
        x = x[mask]; yv = y_np[mask]
        if len(x) == 0:
            rows.append(dict(Feature=SHORT.get(col, col)))
            continue
        mu, sd = float(np.mean(x)), float(np.std(x))
        q01, q99 = float(np.percentile(x, 1)), float(np.percentile(x, 99))
        xmin, xmax = float(np.min(x)), float(np.max(x))
        try:
            r = float(np.corrcoef(x, yv)[0, 1])
        except Exception:
            r = np.nan
        rows.append(dict(Feature=SHORT.get(col, col), Mean=mu, Std=sd, Min=xmin, P1=q01, P99=q99, Max=xmax, Pearson_r=r))
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(SUMMARY_DIR / "feature_summary_compact.csv", index=False)

    disp = df_sum.copy()
    fmt_cols = ["Mean", "Std", "Min", "P1", "P99", "Max", "Pearson_r"]
    for c in fmt_cols:
        disp[c] = disp[c].map(lambda v: "" if pd.isna(v) else f"{v:.3g}")
    col_labels = ["Feature", "Mean", "Std", "Min", "1%", "99%", "Max", "Pearson r"]
    data = [disp["Feature"].tolist()] + [disp[c].tolist() for c in fmt_cols]
    data = list(map(list, zip(*data)))

    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(disp) + 0.5))
    ax.axis("off")
    table = ax.table(cellText=data, colLabels=col_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.1)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.5 if r == 0 else 0.2)
        if r == 0:
            cell.set_facecolor("#f2f2f2")
    plt.tight_layout(pad=0.1)
    plt.savefig(FIG_DIR / "feature_summary_compact.png", dpi=600, bbox_inches='tight')
    plt.close()

def main():
    seed_everything(config.TRAIN_SEED)

    cur = Path(config.CURATED_CSV)
    if not cur.exists():
        raise FileNotFoundError(f"Missing curated snapshot: {cur}")
    df = pd.read_csv(cur)

    need = ["uid", "formula", "E_bar_Npm"] + FEATURES
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Curated CSV missing columns: {missing}")

    X = df[FEATURES].astype(float).copy()
    y = df["E_bar_Npm"].astype(float)
    groups = df["formula"].astype(str).fillna("")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.TRAIN_SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

    scaler = StandardScaler().fit(Xtr.values)
    Xtr_z = scaler.transform(Xtr.values)
    Xte_z = scaler.transform(Xte.values)
    Xtr_z_df = pd.DataFrame(Xtr_z, columns=FEATURES)

    params = dict(config.XGB_PARAMS)
    params["random_state"] = config.TRAIN_SEED
    model = XGBRegressor(**params)
    model.fit(Xtr_z, ytr)

    tr_r2, tr_mae, tr_rmse = _metrics(ytr, model.predict(Xtr_z))
    te_r2, te_mae, te_rmse = _metrics(yte, model.predict(Xte_z))

    gkf = GroupKFold(n_splits=3)
    cv_r2, cv_mae = [], []
    Xnp, ynp, gnp = X.values, y.values, groups.values
    for tr2, va2 in gkf.split(Xnp, ynp, groups=gnp):
        sc = StandardScaler().fit(Xnp[tr2])
        Xtr2z = sc.transform(Xnp[tr2]); Xva2z = sc.transform(Xnp[va2])
        m2 = XGBRegressor(**params); m2.fit(Xtr2z, ynp[tr2])
        pv = m2.predict(Xva2z)
        cv_r2.append(r2_score(ynp[va2], pv))
        cv_mae.append(mean_absolute_error(ynp[va2], pv))

    import joblib, sklearn, xgboost
    joblib.dump(scaler, config.SCALER_PATH) 
    model.save_model(MODELS_DIR / "xgb_model.json")

    _save_json({
        "split": "GroupShuffleSplit by formula (test_size=0.20)",
        "features": FEATURES,
        "n_rows": int(len(df)),
        "n_features": len(FEATURES),
        "train_R2": tr_r2, "train_MAE": tr_mae, "train_RMSE": tr_rmse,
        "test_R2": te_r2, "test_MAE": te_mae, "test_RMSE": te_rmse,
        "cv3_grouped_R2_mean": float(np.mean(cv_r2)), "cv3_grouped_R2_std": float(np.std(cv_r2)),
        "cv3_grouped_MAE_mean": float(np.mean(cv_mae)), "cv3_grouped_MAE_std": float(np.std(cv_mae)),
        "used_best_params": True,
        "params": params,
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
    }, METRICS_DIR / "metrics_xgb.json")

    plt.figure(figsize=(5.6, 5.6))
    yte_pred = model.predict(Xte_z)
    plt.scatter(np.asarray(yte), np.asarray(yte_pred), s=18, alpha=0.85,
                color=SCATTER, edgecolor='black', linewidth=0.5, label="Test points")
    lo = float(min(np.min(yte), np.min(yte_pred)))
    hi = float(max(np.max(yte), np.max(yte_pred)))
    plt.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=IDEAL, label="Perfect (y=x)")
    z = np.polyfit(np.asarray(yte), np.asarray(yte_pred), 1)
    xline = np.linspace(lo, hi, 200)
    yline = z[0] * xline + z[1]
    plt.plot(xline, yline, lw=1.8, color=TREND, label=f"Trend (slope={z[0]:.2f})")
    txt = f"R$^2$ = {te_r2:.3f}\nMAE = {te_mae:.2f} N/m\nRMSE = {te_rmse:.2f} N/m"
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                   va="top", ha="left",
                   bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=TREND, alpha=0.9))
    plt.xlabel(r"True $\bar{\mathrm{E}}$ (N/m)")
    plt.ylabel(r"Predicted $\bar{\mathrm{E}}$ (N/m)")
    plt.title("Prediction parity")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "parity_xgb.png", dpi=600)
    plt.close()

    gain = getattr(model, "feature_importances_", None)
    if gain is not None:
        total_gain = np.sum(gain)
        perc = 100 * gain / total_gain if total_gain > 0 else gain
        order = np.argsort(perc)[::-1]
        labels = [SHORT.get(FEATURES[i], FEATURES[i]) for i in order]
        vals = perc[order]
        plt.figure(figsize=(8.4, 4.8))
        y_pos = np.arange(len(labels))
        bars = plt.barh(y_pos, vals, color=ACCENT, alpha=0.9, edgecolor='black', linewidth=0.3)
        plt.gca().invert_yaxis()
        plt.yticks(y_pos, labels)
        plt.xlabel("Importance (%)")
        plt.title("Feature importance (gain)")
        for bar, val in zip(bars, vals):
            plt.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
                     va='center', ha='right', color='white')
        plt.tight_layout()
        plt.savefig(FIG_DIR / "importance_xgb.png", dpi=600)
        plt.close()

    explainer = shap.TreeExplainer(model)
    Xsub = Xtr_z_df.sample(n=min(4000, len(Xtr_z_df)), random_state=config.TRAIN_SEED)
    sv = explainer.shap_values(Xsub)
    shap.summary_plot(sv, Xsub, show=False)
    plt.title("SHAP summary")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_xgb.png", dpi=600)
    plt.close()

    for FEAT, FNAME in [
        ("MagpieData mean NUnfilled", "shap_dependence_meanNUnfilled.png"),
        ("MagpieData maximum NdValence", "shap_dependence_maxNdValence.png"),
    ]:
        shap.dependence_plot(FEAT, sv, Xsub, interaction_index="MagpieData mean MeltingT", show=False)
        plt.title(f"SHAP dependence — {SHORT.get(FEAT, FEAT)}")
        cbar = plt.gcf().axes[-1] if len(plt.gcf().axes) > 1 else None
        if cbar:
            cbar.set_ylabel("Mean melting T (z-scaled)", rotation=270, labelpad=18)
        plt.tight_layout()
        plt.savefig(FIG_DIR / FNAME, dpi=600)
        plt.close()

    plot_E_histogram(y)
    plot_feature_histograms(X)
    make_feature_summary(X, y)

    print("\n=== Results (grouped by formula) ===")
    print(f"train: R2={tr_r2:.3f}, MAE={tr_mae:.2f}, RMSE={tr_rmse:.2f}")
    print(f"test : R2={te_r2:.3f}, MAE={te_mae:.2f}, RMSE={te_rmse:.2f}")
    print(f"cv3  : R2={np.mean(cv_r2):.3f}±{np.std(cv_r2):.3f}, MAE={np.mean(cv_mae):.2f}±{np.std(cv_mae):.2f}")
    print(f"Figures & files in: {OUT_DIR}")

if __name__ == "__main__":
    main()
