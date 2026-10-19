import gc
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, rankdata, spearmanr
from scipy.spatial import cKDTree
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
y_f = y.astype(np.float64)
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

e1_p = to_probit_split(full_df['ext_1_hybrid'].values, N_tr)
e2_p = to_probit_split(full_df['EXT_SOURCE_2'].values, N_tr)
e3_p = to_probit_split(full_df['ext_3_hybrid'].values, N_tr)
e_mat = np.column_stack([e1_p, e2_p, e3_p])
full_df['bureau_min_deficit'] = to_probit_split(np.min(e_mat, axis=1) - np.mean(e_mat, axis=1), N_tr)
full_df['bureau_spread'] = to_probit_split(np.max(e_mat, axis=1) - np.min(e_mat, axis=1), N_tr)

anchor_pool_dir = sandbox_dir / "core_features"
if not anchor_pool_dir.exists():
    anchor_pool_dir = Path(r"C:\Users\User\Documents\Personal Stuffs\GCI World\output")

anchor_cols = [
    'ext_1_hybrid', 'EXT_SOURCE_2', 'ext_3_hybrid', 'logit_ext_mean',
    'payment_rate', 'annuity_to_income', 'credit_to_income', 'goods_to_credit', 'merton_buffer',
    'bureau_min_deficit', 'bureau_spread', 'age_years', 'employed_years'
]

for pkl_name in ["accepted_subbin_features_full.pkl", "accepted_actuarial_features_full.pkl", "accepted_v5_discrete_tariffs_full.pkl", "accepted_v6_osw_bipartite_full.pkl"]:
    fp = anchor_pool_dir / pkl_name
    if fp.exists():
        d_sub = pd.read_pickle(fp)
        for c in d_sub.columns:
            full_df[c] = d_sub[c].values
            anchor_cols.append(c)

anchor_cols = list(dict.fromkeys(anchor_cols))
X_anchor_tr = full_df.iloc[:N_tr][anchor_cols].copy()

p_base_oof = 1.0 / (1.0 + np.exp(-base_init_logits[:N_tr]))
r_base_oof = y_f[:N_tr] - p_base_oof

man_feature_pool = [
    'ext_1_hybrid', 'EXT_SOURCE_2', 'ext_3_hybrid', 'logit_ext_mean',
    'payment_rate', 'annuity_to_income', 'credit_to_income', 'goods_to_credit',
    'merton_buffer', 'bureau_min_deficit', 'bureau_spread', 'age_years', 'employed_years'
]
if 'cand_tariff_joint_ext_lo_ortho' in full_df.columns:
    man_feature_pool.append('cand_tariff_joint_ext_lo_ortho')
if 'cand_bipartite_2hop_pr_surplus' in full_df.columns:
    man_feature_pool.append('cand_bipartite_2hop_pr_surplus')

Z_raw_man = np.column_stack([to_probit_split(full_df[c].values, N_tr) for c in man_feature_pool])

def haar_orthogonal_directions(k, n_blocks, seed=42):
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_blocks):
        G = rng.standard_normal((k, k))
        Q, R = np.linalg.qr(G)
        d = np.diag(R)
        Q = Q * np.where(d >= 0, 1.0, -1.0)
        blocks.append(Q)
    return np.vstack(blocks)

def mean_w1_to_quantiles(P_eval, q_grid):
    Q_len, L_len = q_grid.shape
    total = np.zeros(P_eval.shape[0], dtype=np.float64)
    for l_idx in range(L_len):
        ql = q_grid[:, l_idx]
        xl = P_eval[:, l_idx]
        pref = np.concatenate([[0.0], np.cumsum(ql)])
        m = np.searchsorted(ql, xl, side='right')
        total += ((2.0 * m - Q_len) * xl - 2.0 * pref[m] + pref[Q_len]) / float(Q_len)
    return total / float(L_len)

tau_grid = np.linspace(1.0 / 32.0, 31.0 / 32.0, 31)

def run_residual_probe(feat_series):
    v = np.asarray(feat_series, dtype=np.float64)
    v_clean = np.where(np.isnan(v), np.nanmedian(v), v)
    df_eval = pd.DataFrame({'cand': v_clean[:N_tr]})
    
    oof_res = np.zeros(N_tr)
    fold_aucs = np.zeros(10)
    params = {
        'objective': 'binary', 'learning_rate': 0.015, 'n_estimators': 30,
        'num_leaves': 6, 'max_depth': 2, 'min_child_samples': 500,
        'colsample_bytree': 1.0, 'subsample': 0.85, 'subsample_freq': 1,
        'reg_alpha': 15.0, 'reg_lambda': 120.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    for f_idx, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(df_eval.iloc[tr_idx], label=y[tr_idx], init_score=base_init_logits[tr_idx])
        bst = lgb.train(params, dtr)
        raw_margin = base_init_logits[va_idx] + bst.predict(df_eval.iloc[va_idx], raw_score=True)
        oof_res[va_idx] = raw_margin
        fold_aucs[f_idx] = roc_auc_score(y[va_idx], raw_margin)
    return roc_auc_score(y, oof_res), fold_aucs

control_auc = roc_auc_score(y, base_init_logits)
control_fold_aucs = np.zeros(10)
for f_idx, (_, va_idx) in enumerate(folds):
    control_fold_aucs[f_idx] = roc_auc_score(y[va_idx], base_init_logits[va_idx])

print(f"Loaded Laptop Manifold Environment: 10-Fold Champion Control AUC = {control_auc:.6f}")
print(f"Monitoring {len(anchor_cols)} active core features in the orthogonality reference set.")
print("Starting Autonomous Klein Hyperbolic & Manifold Search Loop...\n", flush=True)

loop_count = 0
accepted_count = 0

while True:
    loop_count += 1
    t0 = time.time()
    seed = int(np.random.randint(1000, 999999))
    rng = np.random.default_rng(seed)

    generator_mode = rng.choice(["sliced_klein_w1", "shrunk_knn_residual", "conformal_subspace_contrast"])
    cand_vec = None
    cand_name = None
    recipe_desc = ""

    k_sub_size = int(rng.choice([5, 7, 9, 11]))
    sub_indices = rng.choice(Z_raw_man.shape[1], size=k_sub_size, replace=False)
    w_weights = rng.uniform(0.5, 2.0, size=k_sub_size)
    w_weights /= np.sum(w_weights)
    Z_selected = Z_raw_man[:, sub_indices] * np.sqrt(w_weights)[None, :]

    alpha_k = float(rng.uniform(0.12, 0.28))
    e_norm = np.linalg.norm(Z_selected, axis=1, keepdims=True)
    r_poincare = np.tanh(alpha_k * (e_norm / np.sqrt(k_sub_size)))
    U_poincare = Z_selected * (r_poincare / (e_norm + 1e-9))

    if generator_mode == "sliced_klein_w1":
        n_proj = int(rng.choice([8, 16, 24]))
        V_proj = haar_orthogonal_directions(k_sub_size, n_blocks=max(1, n_proj // k_sub_size), seed=seed)[:n_proj]
        P_eval = U_poincare @ V_proj.T
        w1_margin = np.zeros(N_all, dtype=np.float64)
        for tr_idx, va_idx in folds:
            idx_s = tr_idx[y[tr_idx] == 0]
            idx_d = tr_idx[y[tr_idx] == 1]
            q_s = np.quantile(P_eval[idx_s], tau_grid, axis=0)
            q_d = np.quantile(P_eval[idx_d], tau_grid, axis=0)
            w1_margin[va_idx] = mean_w1_to_quantiles(P_eval[va_idx], q_s) - mean_w1_to_quantiles(P_eval[va_idx], q_d)
            w1_margin[N_tr:] += 0.1 * (mean_w1_to_quantiles(P_eval[N_tr:], q_s) - mean_w1_to_quantiles(P_eval[N_tr:], q_d))
        cand_vec = to_probit_split(w1_margin, N_tr)
        cand_name = f"cand_klein_w1_{n_proj}d_{seed}"
        recipe_desc = f"Sliced W1 margin in Klein space across {k_sub_size} features ({n_proj} projections)"

    elif generator_mode == "shrunk_knn_residual":
        k_neighbors = int(rng.choice([100, 180, 260]))
        m0_bayes = 50.0
        out_vec = np.zeros(N_all, dtype=np.float64)
        U_tr = U_poincare[:N_tr]
        U_te = U_poincare[N_tr:]
        tree_tr = cKDTree(U_tr, leafsize=64)

        for tr_idx, va_idx in folds:
            tree_fold = cKDTree(U_tr[tr_idx], leafsize=64)
            dists_va, idxs_va = tree_fold.query(U_tr[va_idx], k=k_neighbors, workers=-1)
            weights_va = 1.0 / (dists_va + 1e-4)
            sum_w_va = np.sum(weights_va, axis=1, keepdims=True)
            norm_w_va = weights_va / (sum_w_va + 1e-9)
            raw_k_res = np.sum(norm_w_va * r_base_oof[tr_idx][idxs_va], axis=1)
            out_vec[va_idx] = (k_neighbors * raw_k_res) / (k_neighbors + m0_bayes)

        dists_te, idxs_te = tree_tr.query(U_te, k=k_neighbors, workers=-1)
        weights_te = 1.0 / (dists_te + 1e-4)
        sum_w_te = np.sum(weights_te, axis=1, keepdims=True)
        norm_w_te = weights_te / (sum_w_te + 1e-9)
        raw_k_res_te = np.sum(norm_w_te * r_base_oof[idxs_te], axis=1)
        out_vec[N_tr:] = (k_neighbors * raw_k_res_te) / (k_neighbors + m0_bayes)

        cand_vec = to_probit_split(out_vec, N_tr)
        cand_name = f"cand_klein_shrunk_knn_k{k_neighbors}_{seed}"
        recipe_desc = f"Empirical-Bayes shrunk k-NN residual diffusion (k={k_neighbors}, m0={m0_bayes})"

    else:
        u_sq = np.clip(np.sum(U_poincare ** 2, axis=1), 0.0, 1.0 - 1e-6)
        conformal_factor = 2.0 / (1.0 - u_sq)
        p_contrast = full_df['payment_rate'].values - full_df['goods_to_credit'].values
        cand_vec = to_probit_split(p_contrast * conformal_factor, N_tr)
        cand_name = f"cand_klein_conformal_contrast_{seed}"
        recipe_desc = f"Payment contrast scaled by Klein-Poincare conformal metric factor"

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

    is_accepted = (delta_auc >= 0.000015 and folds_won >= 8 and max_corr < 0.85)

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
            
        print(f"[ACCEPTED #{accepted_count}] {cand_name} | dAUC: {delta_auc:+.6f} | Folds: {folds_won}/10 | MaxCorr: {max_corr:.3f} | File: {out_parquet.name}", flush=True)

        try:
            os.system(f'cd "{sandbox_dir}" && git add "{out_parquet}" "{log_file}" && git commit -m "Auto-add discovered manifold feature {cand_name}" && git push origin main')
        except Exception:
            pass
        
        anchor_cols.append(cand_name)
        X_anchor_tr[cand_name] = cand_vec[:N_tr]
    else:
        print(f"[Screening #{loop_count}] {cand_name[:36]}... | dAUC: {delta_auc:+.6f} | Folds: {folds_won}/10 | MaxCorr: {max_corr:.3f} ({elapsed:.1f}s)", flush=True)

    gc.collect()
