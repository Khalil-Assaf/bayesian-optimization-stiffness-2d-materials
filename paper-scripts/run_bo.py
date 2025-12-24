from __future__ import annotations
import sys, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils import seed_everything, is_forbidden_formula, apply_house_style

LINE = "#4B0082"
DASH = "#666666"
apply_house_style()
warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT_DIR = getattr(config, "OUT_DIR", ROOT / "outputs")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
HISTORY_DIR = OUT_DIR / "histories"
for d in [OUT_DIR, FIG_DIR, TABLE_DIR, HISTORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEEDS = config.SEEDS
SCALER_PATH = config.SCALER_PATH
CURATED_CSV = config.CURATED_CSV

CAP_Z = 1.5
E_THRESH = 300.0
KNN_K = 20
N_INIT = 128
N_ITERS = 200
BATCH = 10_000
RESAMPLE = 8

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

def _fit_member(X: np.ndarray, y: np.ndarray, seed: int):
    kw = dict(config.XGB_PARAMS)
    kw["random_state"] = seed
    m = XGBRegressor(**kw)
    m.fit(X, y)
    return m

def _ensemble(models, X_):
    return np.column_stack([m.predict(X_) for m in models]).mean(axis=1)

def _norm_cdf(x):
    try:
        from scipy.special import ndtr
        return ndtr(x)
    except Exception:
        x = np.asarray(x, dtype=float)
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

def expected_improvement(mu, sigma, y_best, xi=1e-3):
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - y_best - xi) / sigma
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = _norm_cdf(z)
    return (mu - y_best - xi) * cdf + sigma * pdf

def sobol_or_lhs(n, d, low, high, rng):
    try:
        from scipy.stats import qmc
        if (n & (n - 1)) == 0:
            U = qmc.Sobol(d, scramble=True, seed=int(rng.integers(1, 10_000))).random(n)
        else:
            U = qmc.LatinHypercube(d, seed=int(rng.integers(1, 10_000))).random(n)
        return qmc.scale(U, low, high)
    except Exception:
        U = rng.random((n, d)); return low + U * (high - low)

def sample_box(n, low, high, rng):
    U = rng.random((n, low.size)); return low + U * (high - low)

def get_dataset():
    cur = Path(CURATED_CSV)
    if not cur.exists():
        raise FileNotFoundError(f"Missing curated snapshot: {cur}")
    df = pd.read_csv(cur)
    need = ["uid", "formula", "E_bar_Npm"] + FEATURES
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Curated CSV missing columns: {missing}")
    X = df[FEATURES].astype(float).copy()
    y = df["E_bar_Npm"].astype(float)
    return X, y, "uid", df

def run_one_seed(seed: int, X_df: pd.DataFrame, y: pd.Series, key: str, df: pd.DataFrame):
    print(f"\n=== Distance-constrained BO (seed {seed}) ===")
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    X_np = X_df.values.astype(float)
    members = [_fit_member(X_np, y.values, s) for s in [seed, 1337, 2027, 3141]]

    if Path(SCALER_PATH).exists():
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = StandardScaler().fit(X_np)
    Xz = scaler.transform(X_np)

    low, high = Xz.min(axis=0), Xz.max(axis=0)
    pad = 0.05 * (high - low + 1e-12)
    low -= pad; high += pad

    def f_from_z(Z):
        Xcand = scaler.inverse_transform(Z)
        return _ensemble(members, Xcand)

    nn = NearestNeighbors(n_neighbors=KNN_K, metric="euclidean").fit(Xz)

    Z_obs = sobol_or_lhs(N_INIT, X_np.shape[1], low, high, rng)
    d0, _ = nn.kneighbors(Z_obs, n_neighbors=1, return_distance=True)
    Z_obs = Z_obs[d0[:, 0] <= CAP_Z]
    while Z_obs.shape[0] < N_INIT:
        Z_try = sobol_or_lhs(N_INIT - Z_obs.shape[0], X_np.shape[1], low, high, rng)
        d, _ = nn.kneighbors(Z_try, n_neighbors=1, return_distance=True)
        Z_obs = np.vstack([Z_obs, Z_try[d[:, 0] <= CAP_Z]])
    y_obs = f_from_z(Z_obs)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_np.shape[1]), nu=2.5) + WhiteKernel(1e-6)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=seed)

    history = [{"iter": 0, "y_best": float(np.max(y_obs))}]
    best_so_far = history[0]["y_best"]

    for t in range(1, N_ITERS + 1):
        gp.fit(Z_obs, y_obs)

        ok = None
        for _ in range(RESAMPLE):
            Z_cand = sample_box(BATCH, low, high, rng)
            dmin, _ = nn.kneighbors(Z_cand, n_neighbors=1, return_distance=True)
            mask = dmin[:, 0] <= CAP_Z
            if mask.any():
                ok = Z_cand[mask]; break
        if ok is None:
            ok = Z_cand

        mu, sigma = gp.predict(ok, return_std=True)
        ei = expected_improvement(mu, sigma, best_so_far, xi=1e-3)
        z_star = ok[np.argmax(ei):np.argmax(ei) + 1, :]
        y_star = float(f_from_z(z_star)[0])

        Z_obs = np.vstack([Z_obs, z_star])
        y_obs = np.concatenate([y_obs, [y_star]])
        best_so_far = max(best_so_far, y_star)
        history.append({"iter": t, "y_best": best_so_far})

    hist = pd.DataFrame(history)
    hist_csv = HISTORY_DIR / f"bo_history_dc_seed{seed}.csv"
    hist.to_csv(hist_csv, index=False)

    plt.figure(figsize=(6.6, 4.4))
    plt.step(hist["iter"], hist["y_best"], where="post", lw=2.2, color=LINE, label="Best prediction")
    plt.axhline(E_THRESH, ls="--", color=DASH, lw=1.2, dashes=(6, 3), label=f"Threshold = {E_THRESH:.0f} N/m")
    plt.xlabel("Iteration")
    plt.ylabel(r"Best $\bar{\mathrm{E}}$ (N/m)")
    plt.title(f"BO convergence (seed {seed})")
    plt.legend(loc="lower right", frameon=True)
    ymax = float(25.0 * math.ceil((float(np.nanmax(hist["y_best"])) + 5.0) / 25.0))
    plt.ylim(290.0, ymax)
    out_png = FIG_DIR / f"bo_convergence_dc_seed{seed}.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=600); plt.close()

    gp.fit(Z_obs, y_obs)
    order = np.argsort(y_obs)[::-1]
    Z_sorted = Z_obs[order, :]

    Xz_nn = NearestNeighbors(n_neighbors=KNN_K, metric="euclidean").fit(Xz)
    chosen, used_idx = [], set()
    for j in range(Z_sorted.shape[0]):
        z = Z_sorted[j:j + 1, :]
        dists, inds = Xz_nn.kneighbors(z, n_neighbors=KNN_K, return_distance=True)
        for d, irow in zip(dists.ravel(), inds.ravel()):
            if d <= CAP_Z and irow not in used_idx:
                used_idx.add(irow)
                uid = df.iloc[irow][key]
                formula = str(df.iloc[irow].get("formula", ""))
                if is_forbidden_formula(formula, getattr(config, "EXCLUDE_FORMULAS", set())):
                    continue
                x_row = X_np[irow:irow + 1, :]
                E_pred = float(_ensemble(members, x_row)[0])
                _, sig = gp.predict(z, return_std=True)
                chosen.append({"seed": seed, "uid": uid, "formula": formula,
                               "source": "BO_kNN", "E_pred_Npm": E_pred,
                               "gp_sigma": float(sig[0]), "nn_dist_z": float(d)})
                break

    tab = pd.DataFrame(chosen).sort_values("E_pred_Npm", ascending=False)
    if "E_bar_Npm" in df.columns:
        truth = dict(zip(df[key], df["E_bar_Npm"]))
        tab["E_true_Npm"] = tab["uid"].map(truth)
        tab["abs_err"] = (tab["E_pred_Npm"] - tab["E_true_Npm"]).abs()
        tab["rel_err_%"] = 100.0 * tab["abs_err"] / tab["E_true_Npm"].clip(lower=1e-9)

    tab_hi = tab[tab["E_pred_Npm"] >= E_THRESH].copy().sort_values("E_pred_Npm", ascending=False)
    tab_hi.to_csv(TABLE_DIR / f"paper_table_bo_filtered_seed{seed}.csv", index=False)

    rows = list(tab_hi.to_dict(orient="records"))
    used_uids = set(tab_hi["uid"])
    if len(rows) < 15:
        scores = _ensemble(members, X_np)
        order_all = np.argsort(scores)[::-1]
        for irow in order_all:
            uid = df.iloc[irow][key]
            formula = str(df.iloc[irow].get("formula", ""))
            if uid in used_uids:
                continue
            if is_forbidden_formula(formula, getattr(config, "EXCLUDE_FORMULAS", set())):
                continue
            d = float(np.linalg.norm(Xz[irow] - Z_sorted[0])) if Z_sorted.size else float("inf")
            if d <= CAP_Z:
                rows.append({
                    "seed": seed, "uid": uid, "formula": formula,
                    "source": "DatasetTop", "E_pred_Npm": float(scores[irow]),
                    "gp_sigma": np.nan, "nn_dist_z": d,
                    "E_true_Npm": truth.get(uid) if "E_bar_Npm" in df.columns else np.nan
                })
                used_uids.add(uid)
            if len(rows) >= 15:
                break
    pd.DataFrame(rows).head(15).to_csv(TABLE_DIR / f"paper_table_top15_seed{seed}.csv", index=False)

    print(f"Saved: {hist_csv}, {out_png}")
    print(f"Saved: {TABLE_DIR / f'paper_table_bo_filtered_seed{seed}.csv'}")
    print(f"Saved: {TABLE_DIR / f'paper_table_top15_seed{seed}.csv'} (capped to 15)")

def build_union() -> Path:
    files = sorted(set(TABLE_DIR.glob("paper_table_bo_filtered_seed*.csv")))
    if not files:
        raise SystemExit("No per-seed CSVs found to build the union.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "seed" not in df.columns:
            seed = int(Path(f).stem.split("seed")[-1])
            df["seed"] = seed
        frames.append(df)
    cat = pd.concat(frames, ignore_index=True)

    if "formula" in cat.columns:
        mask = ~cat["formula"].astype(str).apply(
            lambda s: is_forbidden_formula(s, getattr(config, "EXCLUDE_FORMULAS", set()))
        )
        cat = cat[mask].copy()

    if not {"uid", "E_pred_Npm"}.issubset(cat.columns):
        raise KeyError("Expected 'uid' and 'E_pred_Npm' in per-seed files.")
    agg = {"formula": "first", "E_true_Npm": "first", "E_pred_Npm": "max", "nn_dist_z": "min", "gp_sigma": "min"}
    base = cat.groupby("uid", as_index=False).agg(agg)

    seeds_map = (cat.groupby("uid")["seed"]
                 .apply(lambda s: ", ".join(map(str, sorted(set(int(x) for x in s if int(x) >= 0))))).rename("seeds"))
    base = base.merge(seeds_map, on="uid", how="left")

    sort_cols = [c for c in ["E_true_Npm", "E_pred_Npm", "nn_dist_z"] if c in base.columns]
    base = base.sort_values(sort_cols, ascending=[False, False, True][:len(sort_cols)])

    out = TABLE_DIR / "union_bo_candidates_dc.csv"
    base.to_csv(out, index=False)
    print(f"Union saved: {out}  (rows={len(base)})")
    return out

def build_final_15(union_path: Path) -> tuple[Path, Path]:
    df = pd.read_csv(union_path)
    need = {"uid", "formula", "E_true_Npm", "E_pred_Npm", "nn_dist_z"}
    if not need.issubset(df.columns):
        raise KeyError(f"Union missing columns {need - set(df.columns)}")

    mask = (df["nn_dist_z"] <= CAP_Z) & (df["E_pred_Npm"] >= E_THRESH) & (df["E_true_Npm"].notna())
    df = df[mask].copy()

    df = df[~df["formula"].astype(str).apply(
        lambda s: is_forbidden_formula(s, getattr(config, "EXCLUDE_FORMULAS", set()))
    )].copy()

    df = df.sort_values(["E_true_Npm", "E_pred_Npm"], ascending=[False, False]).head(15).copy()

    cols_std = ["uid", "formula", "E_true_Npm", "E_pred_Npm", "nn_dist_z", "seeds"]
    for c in ["seeds"]:
        if c not in df.columns:
            df[c] = "" if c == "seeds" else 1
    out_std = TABLE_DIR / "paper_candidates_final.csv"
    df[cols_std].to_csv(out_std, index=False)
    print(f"Final 15 saved:\n  {out_std}")
    return out_std

def plot_zdist_vs_Edft(union_path: Path):
    df = pd.read_csv(union_path)
    if not {"nn_dist_z", "E_true_Npm"}.issubset(df.columns):
        print("Union missing columns for z-distance/DFT plot; skipping.")
        return

    x = pd.to_numeric(df["nn_dist_z"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["E_true_Npm"], errors="coerce").to_numpy()

    plt.figure(figsize=(6.8, 4.6))
    hb = plt.hexbin(x, y, gridsize=24, mincnt=1, linewidths=0.4, edgecolors="black", cmap="Purples")
    counts = hb.get_array()
    maxc = int(np.nanmax(counts)) if counts.size else 1
    boundaries = np.arange(0.5, maxc + 1.5, 1.0)
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import get_cmap
    cmap = get_cmap("Purples", maxc)
    hb.set_cmap(cmap)
    hb.set_norm(BoundaryNorm(boundaries, cmap.N))
    hb.set_clim(0.5, maxc + 0.5)

    plt.axvline(CAP_Z, ls="--", lw=1.1, color=DASH, label=f"Cap = {CAP_Z}")
    plt.xlabel("Nearest-neighbor distance in z-space (std units)")
    plt.ylabel(r"$\mathrm{E}_{\mathrm{DFT}}$ (N/m)")
    plt.title(r"$z$-distance vs $\mathrm{E}_{\mathrm{DFT}}$")
    cb = plt.colorbar(hb); cb.set_label("Count")
    cb.set_ticks(np.arange(1, maxc + 1, 1))
    cb.set_ticklabels([str(i) for i in range(1, maxc + 1)])
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "zdist_vs_Edft.png", dpi=600)
    plt.close()

def main():
    X, y, key, df = get_dataset()
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURES)
    X = X[FEATURES].copy()

    for s in SEEDS:
        run_one_seed(s, X, y, key, df)

    union_path = build_union()
    build_final_15(union_path)
    plot_zdist_vs_Edft(union_path)

if __name__ == "__main__":
    main()
