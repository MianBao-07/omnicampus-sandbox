# omnicampus-sandbox

Autonomous 24/7 background feature discovery sandbox engine for Omnicampus Assignment 1038.

This engine continually generates and tests mathematical feature transformations (quantile ratios, segmented hazard discrepancies, orthogonal subspace projections, solvency curvatures, demographic contrasts) against our **0.77095 ROC-AUC** champion model.

---

## Directory Structure

```
omnicampus-sandbox/
├── laptop_feature_discovery_daemon.py    # Main 24/7 feature discovery daemon
├── laptop_sandbox_base_logits_077095.npz  # Pre-computed log-odds margins from 0.77095 champion
├── data/                                 # Place train/test preprocessed datasets here
│   ├── train_with_oof_impute.pkl
│   └── test_with_oof_impute.pkl
├── accepted_features/                    # Discovered features passing statistical gates (.parquet)
├── discovery_log.jsonl                   # Audit log of every accepted feature transformation
├── requirements.txt
└── README.md
```

---

## Quickstart on Laptop

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy Dataset Files
Place the preprocessed data files into the `data/` folder:
- `data/train_with_oof_impute.pkl`
- `data/test_with_oof_impute.pkl`

*(Note: Data pickle files are excluded by `.gitignore` to keep the repository lightweight).*

### 3. Launch 24/7 Discovery Engine
```bash
python laptop_feature_discovery_daemon.py
```

---

## Statistical Quality Gate

A candidate feature is **only accepted** and persisted if it satisfies all of the following:
1. **Out-of-Fold Gain:** $\Delta\text{AUC} \ge +0.000010$ ($+1.0\text{ bps}$) over the `0.77095` champion model.
2. **Consistency:** Outperforms the champion control on $\ge 7 / 10$ cross-validation folds.
3. **Orthogonality:** Max Spearman rank correlation with existing core feature sets $< 0.85$.
4. **Cleanliness:** Zero `NaN`s, zero target leakage across validation folds.

Accepted features are saved as `.parquet` files under `accepted_features/` with their full formula and cross-validation statistics recorded in `discovery_log.jsonl`.
