This repo is the code I used for my paper "Prioritizing 2D Materials by In-Plane Stiffness Using Composition-Only Machine Learning and DistanceConstrained Bayesian Optimization" about:
1-training an XGBoost model to predict stiffness of 2D materials (from composition features)
2-running Bayesian Optimization (BO) to propose “good” points
3-BUT with a constraint: don’t go too far from real materials (distance constraint), then map back to nearest real material (kNN idea)

It’s basically my “paper scripts” folder, cleaned a bit.

Where is the code:
Main files are in (paper-scripts):

'train_xgb.py' : trains the ML model + saves metrics and model

'run_bo.py' : runs BO (multiple seeds), applies distance filtering, saves candidate tables + plots

'config.py' : parameters, seeds, paths

'utils.py'  : helper functions

Outputs go into 'paper-scripts/outputs/' (tables/figures/histories/models).

Running it:
I used Python 3.10

Install:

''bash
pip install -r requirements.txt
''

Then:

''bash
python paper-scripts/train_xgb.py
python paper-scripts/run_bo.py
''

Dataset (C2DB note): The dataset is derived from C2DB (stiffness values + Magpie composition features).  
Because of redistribution rules, I’m **not** committing the full dataset here by default.

The scripts expect:`paper-scripts/data/Dataset.csv`

So to reproduce everything you need to provide the dataset in that location (from official source / permission).

What you get in outputs:
1- metrics json
2- SHAP/importance plots
3- BO convergence plots (multi-seed)
4- CSV tables of candidates (including the final filtered list)

Some of the outputs are already included because they were used in the paper figures, but you can regenerate them.

Additional notes
1- A lot of results depend on random seeds (I included multiple).
2- The “distance constraint” is there because BO can suggest unrealistic feature combos otherwise.

(If you use it): Please cite C2DB + the original sources. Also feel free to message me if you want help reproducing.
