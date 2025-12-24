from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

CURATED_PATH = DATA_DIR / "Dataset.csv"
CURATED_CSV = CURATED_PATH
TARGET_NAME = "E_bar_Npm"
KEY_NAME = "uid"
FORMULA_COL = "formula"

SEEDS = [6, 68, 6898, 13579, 681998]
TRAIN_SEED = SEEDS[0]

SCALER_PATH = OUT_DIR / "models" / "scaler_train.pkl"

XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=1.0,
    colsample_bytree=0.65,
    reg_lambda=1.0,
    reg_alpha=1.0,
    tree_method="hist",
    n_jobs=-1,
)

EXCLUDE_FORMULAS = set()

def ensure_dirs():
    for p in [DATA_DIR, OUT_DIR]:
        p.mkdir(parents=True, exist_ok=True)

ensure_dirs()
