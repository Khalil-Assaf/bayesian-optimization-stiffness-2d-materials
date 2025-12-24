import os
import random
import numpy as np
from typing import Iterable
import re
import matplotlib as mpl

def seed_everything(seed: int = 6):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

def apply_house_style():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif']  = ['Times New Roman', 'Times', 'DejaVu Serif']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['mathtext.default'] = 'rm'
    mpl.rcParams.update({
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    })

_EARLY_TM = {"Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"}
_TERMINATORS = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te"}

_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

def _parse_formula(formula: str) -> dict:
    comps = {}
    for el, cnt in _ELEM_RE.findall(formula):
        val = float(cnt) if cnt not in ("", None) else 1.0
        comps[el] = comps.get(el, 0.0) + val
    return comps

def _is_bare_mxene(comps: dict) -> bool:
    if not comps:
        return False
    allowed = set(_EARLY_TM) | {"C", "N"}
    if any(el not in allowed for el in comps.keys()):
        return False
    has_tm = any(el in _EARLY_TM for el in comps.keys())
    has_cn = any(el in {"C", "N"} for el in comps.keys())
    return has_tm and has_cn

def _is_h_only_termination(comps: dict) -> bool:
    if "H" not in comps:
        return False
    return not any(t in comps for t in _TERMINATORS)

def is_forbidden_formula(formula: str, hard_block: Iterable[str] = ()) -> bool:
    if not isinstance(formula, str) or not formula:
        return False
    if formula in hard_block:
        return True
    comps = _parse_formula(formula)
    return _is_bare_mxene(comps) or _is_h_only_termination(comps)
