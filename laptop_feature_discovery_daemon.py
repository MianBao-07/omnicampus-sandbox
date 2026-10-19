import gc
import json
import os
import sys
import time
import uuid
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

sandbox_dir = Path(__file__).resolve().parent
output_dir = sandbox_dir / "accepted_features"
output_dir.mkdir(parents=True, exist_ok=True)
log_file = sandbox_dir / "discovery_log.jsonl"

data_dir = sandbox_dir / "data"
if not data_dir.exists():
    data_dir = Path(r"C:\Users\User\Documents\Personal Stuffs\GCI World\input")

train_df = pd.read_pickle(data_dir / "train_with_oof_impute.pkl")
test_df = pd.read_pickle(data_dir / "test_with_oof_impute.pkl")
y = train_df['TARGET'].values.astype(np.int32)
N_tr = len(train_df)
N_te = len(test_df)
N_all = N_tr + N_te

full_df = pd.concat([train_df.drop(columns=['TARGET'], errors='ignore'), test_df], ignore_index=True)

logits_file = sandbox_dir / "laptop_sandbox_base_logits_077095.npz"
if not logits_file.exists():
    logits_file = Path(r"C:\Users\User\Documents\Personal Stuffs\GCI World\output\laptop_sandbox_base_logits_077095.npz")

assert logits_file.exists(), f"Logits file not found at {logits_file}"
logits_data = np.load(logits_file)
base_init_logits = logits_data['base_init_logits']

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
folds = list(skf.split(train_df, y))

def to_probit(p):
    p_arr = np.asarray(p, dtype=np.float64)
    p_clean = np.where(np.isnan(p_arr), np.nanmedian(p_arr), p_arr)
    N = len(p_clean)
    return norm.ppf((rankdata(p_clean, method='average') - 0.375) / (N + 0.25))

def to_probit_split(p, n_tr):
    p_arr = np.asarray(p, dtype=np.float64)
    out = np.empty_like(p_arr, dtype=np.float64)
    out[:n_tr] = to_probit(p_arr[:n_tr])
    out[n_tr:] = to_probit(p_arr[n_tr:])
    return out

full_df['DAYS_EMPLOYED_clean'] = full_df['DAYS_EMPLOYED'].replace({365243: np.nan})
full_df['age_years'] = -full_df['DAYS_BIRTH'] / 365.25
full_df['employed_years'] = -full_df['DAYS_EMPLOYED_clean'] / 365.25
full_df['ext_1_hybrid'] = full_df['EXT_SOURCE_1'].fillna(full_df['EXT_SOURCE_1_PRED'])
full_df['ext_3_hybrid'] = full_df['EXT_SOURCE_3'].fillna(full_df['EXT_SOURCE_3_PRED'])
full_df['payment_rate'] = full_df['AMT_ANNUITY'] / (full_df['AMT_CREDIT'] + 1.0)
full_df['annuity_to_income'] = full_df['AMT_ANNUITY'] / (full_df['AMT_INCOME_TOTAL'] + 1.0)
full_df['credit_to_income'] = full_df['AMT_CREDIT'] / (full_df['AMT_INCOME_TOTAL'] + 1.0)
full_df['goods_to_credit'] = full_df['AMT_GOODS_PRICE'] / (full_df['AMT_CREDIT'] + 1.0)
full_df['merton_buffer'] = (full_df['AMT_INCOME_TOTAL'] - 12.0 * full_df['AMT_ANNUITY']) / (full_df['AMT_CREDIT'] + 1.0)

eps = 1e-4
e2_clean = full_df['EXT_SOURCE_2'].fillna(0.5).clip(eps, 1.0 - eps)
l1 = np.log(full_df['ext_1_hybrid'].clip(eps, 1.0 - eps) / (1.0 - full_df['ext_1_hybrid'].clip(eps, 1.0 - eps)))
l2 = np.log(e2_clean / (1.0 - e2_clean))
l3 = np.log(full_df['ext_3_hybrid'].clip(eps, 1.0 - eps) / (1.0 - full_df['ext_3_hybrid'].clip(eps, 1.0 - eps)))
full_df['logit_ext_mean'] = (l1 + l2 + l3) / 3.0
full_df['log_odds_ext_consensus'] = 0.35 * l1 + 0.35 * l2 + 0.30 * l3

anchor_cols = [
    'ext_1_hybrid', 'EXT_SOURCE_2', 'ext_3_hybrid', 'logit_ext_mean', 'log_odds_ext_consensus',
    'payment_rate', 'annuity_to_income', 'credit_to_income', 'goods_to_credit', 'merton_buffer',
    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED_clean', 'DAYS_ID_PUBLISH'
]

anchor_pool_dir = sandbox_dir / "accepted_features_pool"
if not anchor_pool_dir.exists():
    anchor_pool_dir = Path(r"C:\Users\User\Documents\Personal Stuffs\GCI World\output")

for pkl_name in ["accepted_subbin_features_full.pkl", "accepted_actuarial_features_full.pkl", "accepted_v4_edelberg_ael_full.pkl", "accepted_v5_discrete_tariffs_full.pkl", "accepted_v6_osw_bipartite_full.pkl"]:
    fp = anchor_pool_dir / pkl_name
    if fp.exists():
        d_sub = pd.read_pickle(fp)
        for c in d_sub.columns:
            full_df[c] = d_sub[c].values
            anchor_cols.append(c)

anchor_cols = list(dict.fromkeys(anchor_cols))
X_anchor_tr = full_df.iloc[:N_tr][anchor_cols].copy()

def run_residual_probe(feat_series):
    v = np.asarray(feat_series, dtype=np.float64)
    v_clean = np.where(np.isnan(v), np.nanmedian(v), v)
    df_eval = pd.DataFrame({'cand': v_clean[:N_tr]})
    
    oof_res = np.zeros(N_tr)
    fold_aucs = np.zeros(10)
    params = {
        'objective': 'binary', 'learning_rate': 0.020, 'n_estimators': 80,
        'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 150,
        'colsample_bytree': 1.0, 'subsample': 0.85, 'subsample_freq': 1,
        'reg_alpha': 5.0, 'reg_lambda': 40.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    for f_idx, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(df_eval.iloc[tr_idx], label=y[tr_idx], init_score=base_init_logits[tr_idx])
        dva = lgb.Dataset(df_eval.iloc[va_idx], label=y[va_idx], init_score=base_init_logits[va_idx], reference=dtr)
        bst = lgb.train(params, dtr, valid_sets=[dva], callbacks=[lgb.early_stopping(25, verbose=False)])
        raw_margin = base_init_logits[va_idx] + bst.predict(df_eval.iloc[va_idx], raw_score=True)
        oof_res[va_idx] = raw_margin
        fold_aucs[f_idx] = roc_auc_score(y[va_idx], raw_margin)
    return roc_auc_score(y, oof_res), fold_aucs

control_auc = roc_auc_score(y, base_init_logits)
control_fold_aucs = np.zeros(10)
for f_idx, (_, va_idx) in enumerate(folds):
    control_fold_aucs[f_idx] = roc_auc_score(y[va_idx], base_init_logits[va_idx])

print(f"Loaded Sandbox Environment: 10-Fold Champion Control AUC = {control_auc:.6f}")
print(f"Monitoring {len(anchor_cols)} active core features in the orthogonality reference set.")
print("Starting 24/7 feature discovery loop... (Press Ctrl+C to terminate cleanly)\n")

generator_bank = [
    "cross_term_quantile_ratio",
    "segmented_hazard_discrepancy",
    "bipartite_demographic_residual",
    "orthogonal_subspace_projection",
    "non_linear_solvency_curvature"
]

loop_count = 0
accepted_count = 0

while True:
    loop_count += 1
    t0 = time.time()
    mode = np.random.choice(generator_bank)
    seed = int(np.random.randint(1000, 999999))
    rng = np.random.default_rng(seed)

    cand_name = None
    cand_vec = None
    recipe_desc = ""

    if mode == "cross_term_quantile_ratio":
        col1 = np.random.choice(['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE'])
        col2 = np.random.choice(['DAYS_BIRTH', 'DAYS_EMPLOYED_clean', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION'])
        v1 = full_df[col1].fillna(full_df[col1].median()).values
        v2 = np.abs(full_df[col2].fillna(full_df[col2].median()).values) + 1.0
        p_power = float(np.random.choice([0.5, 0.75, 1.0, 1.33, 1.5]))
        cand_vec = to_probit_split((v1 ** p_power) / v2, N_tr)
        cand_name = f"cand_ratio_{col1}_{col2}_p{int(p_power*100)}_{seed}"
        recipe_desc = f"({col1}**{p_power}) / abs({col2})"

    elif mode == "segmented_hazard_discrepancy":
        seg_col = np.random.choice(['OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE'])
        val_col = np.random.choice(['payment_rate', 'credit_to_income', 'logit_ext_mean', 'merton_buffer'])
        s_val = full_df[val_col].fillna(full_df[val_col].median())
        m = float(np.random.choice([10.0, 25.0, 50.0]))
        g_mean = full_df.groupby(seg_col)[val_col].transform('mean').fillna(s_val.mean()).values
        g_std = full_df.groupby(seg_col)[val_col].transform('std').fillna(1.0).replace(0, 1.0).values
        cand_vec = to_probit_split((s_val.values - g_mean) / (g_std + 1e-4), N_tr)
        cand_name = f"cand_seg_zscore_{seg_col}_{val_col}_{seed}"
        recipe_desc = f"z-score of {val_col} within {seg_col}"

    elif mode == "orthogonal_subspace_projection":
        sub_pool = ['ext_1_hybrid', 'EXT_SOURCE_2', 'ext_3_hybrid', 'payment_rate', 'credit_to_income', 'goods_to_credit', 'merton_buffer', 'age_years', 'employed_years']
        k_pick = np.random.choice(sub_pool, size=min(4, len(sub_pool)), replace=False)
        Z_sub = np.column_stack([to_probit_split(full_df[c].fillna(full_df[c].median()).values, N_tr) for c in k_pick])
        G = rng.standard_normal((Z_sub.shape[1], 1))
        G = G / (np.linalg.norm(G) + 1e-9)
        proj = (Z_sub @ G).flatten()
        cand_vec = to_probit_split(proj, N_tr)
        cand_name = f"cand_subspace_proj_{len(k_pick)}d_{seed}"
        recipe_desc = f"Random Haar orthogonal 1D projection across {list(k_pick)}"

    elif mode == "non_linear_solvency_curvature":
        base_s = full_df['merton_buffer'].fillna(0.0).values
        gate_s = to_probit_split(full_df['logit_ext_mean'].values, N_tr)
        k_deg = float(np.random.choice([0.3, 0.5, 0.7, 1.2, 1.5]))
        cand_vec = to_probit_split(np.sign(base_s) * (np.abs(base_s) ** k_deg) * np.tanh(gate_s), N_tr)
        cand_name = f"cand_solvency_curvature_k{int(k_deg*100)}_{seed}"
        recipe_desc = f"sign(merton_buffer)*abs(merton_buffer)**{k_deg} * tanh(logit_ext)"

    else:
        v1 = to_probit_split(full_df['payment_rate'].values, N_tr)
        v2 = to_probit_split(full_df['goods_to_credit'].values, N_tr)
        alpha = float(np.random.uniform(0.2, 0.8))
        cand_vec = to_probit_split(alpha * v1 - (1.0 - alpha) * v2, N_tr)
        cand_name = f"cand_bipartite_demographic_{seed}"
        recipe_desc = f"Linear contrast {alpha:.2f}*payment_rate - {1.0-alpha:.2f}*goods_to_credit"

    if cand_vec is None or np.isnan(cand_vec).any():
        continue

    v_tr = cand_vec[:N_tr]
    max_corr = 0.0
    for a in np.random.choice(anchor_cols, size=min(12, len(anchor_cols)), replace=False):
        c_val = abs(spearmanr(v_tr, X_anchor_tr[a].fillna(0).values).correlation)
        if c_val > max_corr:
            max_corr = c_val

    if max_corr >= 0.85:
        continue

    cand_auc, fold_aucs = run_residual_probe(cand_vec)
    delta_auc = cand_auc - control_auc
    folds_won = int(np.sum(fold_aucs > control_fold_aucs))
    elapsed = time.time() - t0

    is_accepted = (delta_auc >= 0.000010 and folds_won >= 7 and max_corr < 0.85)

    if is_accepted:
        accepted_count += 1
        out_parquet = output_dir / f"{cand_name}.parquet"
        feat_df = pd.DataFrame({
            'SK_ID_CURR': full_df['SK_ID_CURR'].values,
            cand_name: cand_vec
        })
        feat_df.to_parquet(out_parquet, index=False)
        
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feature_name": cand_name,
            "recipe": recipe_desc,
            "delta_auc": round(delta_auc, 6),
            "folds_won": folds_won,
            "max_corr": round(max_corr, 4),
            "file": str(out_parquet.name),
            "eval_duration_sec": round(elapsed, 2)
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        print(f"[ACCEPTED #{accepted_count}] {cand_name} | dAUC: {delta_auc:+.6f} | Folds: {folds_won}/10 | MaxCorr: {max_corr:.3f} | File: {out_parquet.name}")
        sys.stdout.flush()
        
        full_df[cand_name] = cand_vec
        anchor_cols.append(cand_name)
        X_anchor_tr[cand_name] = cand_vec[:N_tr]
    else:
        if loop_count % 10 == 0:
            print(f"[Screening #{loop_count}] Last: {cand_name[:32]}... | dAUC: {delta_auc:+.6f} | Folds: {folds_won}/10 | MaxCorr: {max_corr:.3f} ({elapsed:.1f}s)")
            sys.stdout.flush()

    gc.collect()
