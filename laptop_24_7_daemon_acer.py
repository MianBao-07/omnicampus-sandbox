import os
import sys
import gc
import time
import hashlib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, rankdata
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

t_global_start = time.time()
base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)
data_dir = base_dir / "data"

matrix_cache = output_dir / "matrix_226_features_buhlmann_tier2.npz"
if not matrix_cache.exists():
    matrix_cache = data_dir / "matrix_226_features_buhlmann_tier2.npz"
if not matrix_cache.exists():
    matrix_cache = base_dir / "matrix_226_features_buhlmann_tier2.npz"
if not matrix_cache.exists():
    matrix_cache = Path(r"C:\Users\User\Documents\Personal Stuffs\GCI World\output\matrix_226_features_buhlmann_tier2.npz")
if not matrix_cache.exists():
    raise FileNotFoundError("matrix_226_features_buhlmann_tier2.npz not found in output/, data/, or root.")

cpu_total = os.cpu_count() or 4
worker_threads = max(2, cpu_total - 2)

print(f"==================================================", flush=True)
print(f"ACER ASPIRE A515-45 24/7 AUTONOMOUS BAGGING DAEMON", flush=True)
print(f"Detected {cpu_total} Logical Cores | Throttled to {worker_threads} Threads for Quiet & Cool 24/7 Run", flush=True)
print(f"==================================================", flush=True)

print(f"Loading pre-computed 226-feature matrix from {matrix_cache.name}...", flush=True)
d_mat = np.load(matrix_cache, allow_pickle=True)
X_tr = d_mat['X_tr']
X_te = d_mat['X_te']
y = d_mat['y_tr']
sk_id_curr_test = d_mat['sk_id_curr_test']

N_tr, D = X_tr.shape
N_te = X_te.shape[0]
print(f"Dataset Loaded: {N_tr} train, {N_te} test, {D} features in {time.time()-t_global_start:.2f}s", flush=True)

seeds = [100, 2024, 777, 314, 999, 1234, 555, 888, 4242, 9999]

oof_lgb_cons_bag = np.zeros(N_tr, dtype=np.float64)
test_lgb_cons_bag = np.zeros(N_te, dtype=np.float64)
oof_lgb_et_bag = np.zeros(N_tr, dtype=np.float64)
test_lgb_et_bag = np.zeros(N_te, dtype=np.float64)
oof_xgb_bag = np.zeros(N_tr, dtype=np.float64)
test_xgb_bag = np.zeros(N_te, dtype=np.float64)
oof_cat_bag = np.zeros(N_tr, dtype=np.float64)
test_cat_bag = np.zeros(N_te, dtype=np.float64)

checkpoint_file = output_dir / "laptop_24_7_acer_checkpoint.npz"
completed_seeds = []

if checkpoint_file.exists():
    try:
        cp = np.load(checkpoint_file)
        completed_seeds = list(cp['completed_seeds'])
        oof_lgb_cons_bag = cp['oof_lgb_cons_bag']
        test_lgb_cons_bag = cp['test_lgb_cons_bag']
        oof_lgb_et_bag = cp['oof_lgb_et_bag']
        test_lgb_et_bag = cp['test_lgb_et_bag']
        oof_xgb_bag = cp['oof_xgb_bag']
        test_xgb_bag = cp['test_xgb_bag']
        oof_cat_bag = cp['oof_cat_bag']
        test_cat_bag = cp['test_cat_bag']
        print(f"Resuming from checkpoint: {len(completed_seeds)} seeds already completed ({completed_seeds})", flush=True)
    except Exception as e:
        print(f"Could not load checkpoint: {e}", flush=True)

for seed_idx, seed in enumerate(seeds):
    if seed in completed_seeds:
        continue

    t_seed_start = time.time()
    print(f"\n--------------------------------------------------", flush=True)
    print(f"STARTING SEED {seed} ({len(completed_seeds)+1}/{len(seeds)})", flush=True)
    print(f"--------------------------------------------------", flush=True)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = list(skf.split(X_tr, y))

    oof_lgb_cons_s = np.zeros(N_tr, dtype=np.float64)
    test_lgb_cons_s = np.zeros(N_te, dtype=np.float64)
    oof_lgb_et_s = np.zeros(N_tr, dtype=np.float64)
    test_lgb_et_s = np.zeros(N_te, dtype=np.float64)
    oof_xgb_s = np.zeros(N_tr, dtype=np.float64)
    test_xgb_s = np.zeros(N_te, dtype=np.float64)
    oof_cat_s = np.zeros(N_tr, dtype=np.float64)
    test_cat_s = np.zeros(N_te, dtype=np.float64)

    lgb_cons_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.025,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 80,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'reg_alpha': 2.0,
        'reg_lambda': 12.0,
        'random_state': seed,
        'n_jobs': worker_threads,
        'verbose': -1
    }

    lgb_et_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'extra_trees': True,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'min_child_samples': 70,
        'subsample': 0.85,
        'colsample_bytree': 0.45,
        'reg_alpha': 1.0,
        'reg_lambda': 20.0,
        'random_state': seed + 500,
        'n_jobs': worker_threads,
        'verbose': -1
    }

    xgb_params = {
        'n_estimators': 1500,
        'learning_rate': 0.025,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.4,
        'min_child_weight': 30,
        'reg_lambda': 15.0,
        'reg_alpha': 2.0,
        'tree_method': 'hist',
        'device': 'cpu',
        'eval_metric': 'auc',
        'early_stopping_rounds': 100,
        'random_state': seed,
        'n_jobs': worker_threads
    }

    cat_params = {
        'iterations': 1200,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 15.0,
        'rsm': 0.5,
        'eval_metric': 'AUC',
        'random_seed': seed,
        'od_type': 'Iter',
        'od_wait': 80,
        'task_type': 'CPU',
        'thread_count': worker_threads,
        'verbose': 0
    }

    print(f"--- Training LGBM Cons (Seed {seed}) ---", flush=True)
    for f_i, (tr_i, va_i) in enumerate(folds):
        dtr = lgb.Dataset(X_tr[tr_i], label=y[tr_i])
        dva = lgb.Dataset(X_tr[va_i], label=y[va_i], reference=dtr)
        m = lgb.train(lgb_cons_params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(80, verbose=False)])
        oof_lgb_cons_s[va_i] = m.predict(X_tr[va_i])
        test_lgb_cons_s += m.predict(X_te) / 10.0
    print(f"  LGBM Cons Seed {seed} OOF: {roc_auc_score(y, oof_lgb_cons_s):.5f}", flush=True)

    print(f"--- Training LGBM Extra-Trees (Seed {seed}) ---", flush=True)
    for f_i, (tr_i, va_i) in enumerate(folds):
        dtr = lgb.Dataset(X_tr[tr_i], label=y[tr_i])
        dva = lgb.Dataset(X_tr[va_i], label=y[va_i], reference=dtr)
        m = lgb.train(lgb_et_params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(80, verbose=False)])
        oof_lgb_et_s[va_i] = m.predict(X_tr[va_i])
        test_lgb_et_s += m.predict(X_te) / 10.0
    print(f"  LGBM ET Seed {seed} OOF: {roc_auc_score(y, oof_lgb_et_s):.5f}", flush=True)

    print(f"--- Training XGBoost Hist (Seed {seed}) ---", flush=True)
    for f_i, (tr_i, va_i) in enumerate(folds):
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X_tr[tr_i], y[tr_i], eval_set=[(X_tr[va_i], y[va_i])], verbose=False)
        oof_xgb_s[va_i] = m.predict_proba(X_tr[va_i])[:, 1]
        test_xgb_s += m.predict_proba(X_te)[:, 1] / 10.0
    print(f"  XGB Hist Seed {seed} OOF: {roc_auc_score(y, oof_xgb_s):.5f}", flush=True)

    print(f"--- Training CatBoost Symmetric (Seed {seed}) ---", flush=True)
    for f_i, (tr_i, va_i) in enumerate(folds):
        m = CatBoostClassifier(**cat_params)
        m.fit(X_tr[tr_i], y[tr_i], eval_set=(X_tr[va_i], y[va_i]), use_best_model=True)
        oof_cat_s[va_i] = m.predict_proba(X_tr[va_i])[:, 1]
        test_cat_s += m.predict_proba(X_te)[:, 1] / 10.0
    print(f"  CatBoost Seed {seed} OOF: {roc_auc_score(y, oof_cat_s):.5f}", flush=True)

    oof_lgb_cons_bag += oof_lgb_cons_s
    test_lgb_cons_bag += test_lgb_cons_s
    oof_lgb_et_bag += oof_lgb_et_s
    test_lgb_et_bag += test_lgb_et_s
    oof_xgb_bag += oof_xgb_s
    test_xgb_bag += test_xgb_s
    oof_cat_bag += oof_cat_s
    test_cat_bag += test_cat_s

    completed_seeds.append(seed)
    np.savez_compressed(
        checkpoint_file,
        completed_seeds=np.array(completed_seeds),
        oof_lgb_cons_bag=oof_lgb_cons_bag,
        test_lgb_cons_bag=test_lgb_cons_bag,
        oof_lgb_et_bag=oof_lgb_et_bag,
        test_lgb_et_bag=test_lgb_et_bag,
        oof_xgb_bag=oof_xgb_bag,
        test_xgb_bag=test_xgb_bag,
        oof_cat_bag=oof_cat_bag,
        test_cat_bag=test_cat_bag
    )
    print(f"Seed {seed} finished in {time.time()-t_seed_start:.1f}s. Checkpoint saved to disk.", flush=True)

n_done = len(completed_seeds)
oof_lgb_cons_final = oof_lgb_cons_bag / n_done
test_lgb_cons_final = test_lgb_cons_bag / n_done
oof_lgb_et_final = oof_lgb_et_bag / n_done
test_lgb_et_final = test_lgb_et_bag / n_done
oof_xgb_final = oof_xgb_bag / n_done
test_xgb_final = test_xgb_bag / n_done
oof_cat_final = oof_cat_bag / n_done
test_cat_final = test_cat_bag / n_done

print(f"\n==================================================", flush=True)
print(f"LAPTOP 24/7 BAGGED PILLARS BENCHMARK ({n_done} SEEDS)", flush=True)
print(f"==================================================", flush=True)
print(f"Bagged LGBM Conservative OOF AUC: {roc_auc_score(y, oof_lgb_cons_final):.6f}", flush=True)
print(f"Bagged LGBM Extra-Trees  OOF AUC: {roc_auc_score(y, oof_lgb_et_final):.6f}", flush=True)
print(f"Bagged XGBoost Hist      OOF AUC: {roc_auc_score(y, oof_xgb_final):.6f}", flush=True)
print(f"Bagged CatBoost          OOF AUC: {roc_auc_score(y, oof_cat_final):.6f}", flush=True)

def to_probit(p):
    p_arr = np.asarray(p, dtype=np.float64)
    p_clean = np.where(np.isnan(p_arr), np.nanmedian(p_arr), p_arr)
    N = len(p_clean)
    return norm.ppf((rankdata(p_clean, method='average') - 0.375) / (N + 0.25))

z_lgb_cons = to_probit(oof_lgb_cons_final)
z_lgb_et = to_probit(oof_lgb_et_final)
z_xgb = to_probit(oof_xgb_final)
z_cat = to_probit(oof_cat_final)

zt_lgb_cons = to_probit(test_lgb_cons_final)
zt_lgb_et = to_probit(test_lgb_et_final)
zt_xgb = to_probit(test_xgb_final)
zt_cat = to_probit(test_cat_final)

streams_oof = [z_lgb_cons, z_lgb_et, z_xgb, z_cat]
streams_test = [zt_lgb_cons, zt_lgb_et, zt_xgb, zt_cat]
w_init = [0.45, 0.25, 0.15, 0.15]

cloud_paths = [
    output_dir / "oof_and_test_cloud_burst_5seed.npz",
    base_dir / "oof_and_test_cloud_burst_5seed.npz"
]
cloud_file = None
for cp in cloud_paths:
    if cp.exists():
        cloud_file = cp
        break

if cloud_file:
    try:
        d_b17 = np.load(cloud_file)
        streams_oof.append(to_probit(d_b17['oof_xgb_cuda']))
        streams_test.append(to_probit(d_b17['test_xgb_cuda']))
        streams_oof.append(to_probit(d_b17['oof_cb_lossguide']))
        streams_test.append(to_probit(d_b17['test_cb_lossguide']))
        w_init = [0.40, 0.15, 0.10, 0.15, 0.12, 0.08]
        print(f"Integrated anchor cloud streams from {cloud_file.name}", flush=True)
    except Exception as e:
        print(f"Notice loading cloud anchor: {e}", flush=True)

stream_matrix_oof = np.column_stack(streams_oof)
stream_matrix_test = np.column_stack(streams_test)

def loss_func(w):
    w_norm = np.maximum(0, w)
    w_norm = w_norm / (np.sum(w_norm) + 1e-9)
    blend = stream_matrix_oof @ w_norm
    return -roc_auc_score(y, blend)

opt = minimize(loss_func, np.array(w_init), method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
w_opt = np.maximum(0, opt.x)
w_opt /= np.sum(w_opt)

z_anchor_oof = stream_matrix_oof @ w_opt
z_anchor_test = stream_matrix_test @ w_opt

auc_anchor = roc_auc_score(y, z_anchor_oof)
print(f"\n==========================================", flush=True)
print(f"LAPTOP 24/7 BAGGED {n_done}-SEED SIMPLEX AUC: {auc_anchor:.6f}", flush=True)
print(f"Gain over Champion B21 (0.775996): +{(auc_anchor - 0.775996)*10000:+.2f} bps ({auc_anchor - 0.775996:+.6f})", flush=True)
print(f"Optimal Simplex Weights: {np.round(w_opt, 4).tolist()}", flush=True)
print(f"==========================================", flush=True)

rank_test = rankdata(z_anchor_test, method='average') / (len(z_anchor_test) + 1.0)
final_probs = expit(norm.ppf(rank_test.clip(1e-6, 1.0 - 1e-6)))

sub_df = pd.DataFrame({
    'SK_ID_CURR': sk_id_curr_test,
    'TARGET': final_probs
})

b_out = output_dir / f"submission_emergency_backup_22_laptop_acer_{n_done}seed_{auc_anchor:.6f}.csv"
sub_df.to_csv(b_out, index=False)
h_out = hashlib.sha256(b_out.read_bytes()).hexdigest()

print(f"\nLocked Candidate Artifact: {b_out.name}", flush=True)
print(f"Candidate SHA-256: {h_out}", flush=True)

if auc_anchor > 0.775996:
    active_sub_out = output_dir / "submission.csv"
    sub_df.to_csv(active_sub_out, index=False)
    h_active = hashlib.sha256(active_sub_out.read_bytes()).hexdigest()
    print(f"Active Submission updated to {auc_anchor:.6f} CV: {h_active == h_out}", flush=True)
else:
    print(f"Candidate did not exceed champion 0.775996; active submission preserved.", flush=True)

np.savez_compressed(
    output_dir / f"oof_and_test_laptop_acer_zoo_{n_done}seed.npz",
    oof_lgb_cons=oof_lgb_cons_final, test_lgb_cons=test_lgb_cons_final,
    oof_lgb_et=oof_lgb_et_final, test_lgb_et=test_lgb_et_final,
    oof_xgb=oof_xgb_final, test_xgb=test_xgb_final,
    oof_cat=oof_cat_final, test_cat=test_cat_final,
    y_true=y
)

print(f"Total Laptop Daemon Runtime: {time.time()-t_global_start:.1f}s", flush=True)
