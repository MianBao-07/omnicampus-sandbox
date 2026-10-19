import os
import gc
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

start_time = time.time()
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
core_feat_dir = base_dir / "core_features"
accepted_dir = base_dir / "accepted_features"
output_dir = base_dir / "burst_output"
output_dir.mkdir(parents=True, exist_ok=True)

train_path = data_dir / "train_with_oof_impute.pkl"
test_path = data_dir / "test_with_oof_impute.pkl"

if not train_path.exists():
    print(f"Error: {train_path} not found. Place dataset files in data/ directory.")
    sys.exit(1)

train_df = pd.read_pickle(train_path)
test_df = pd.read_pickle(test_path)
N_tr = len(train_df)
N_te = len(test_df)
N_all = N_tr + N_te

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

combined = pd.concat([train_df.drop(columns=['TARGET'], errors='ignore'), test_df], ignore_index=True)
combined['client_key_1'] = combined['DAYS_BIRTH'].astype(str) + '_' + combined['CODE_GENDER'].astype(str) + '_' + combined['DAYS_REGISTRATION'].astype(str)
combined['client_key_2'] = combined['DAYS_BIRTH'].astype(str) + '_' + combined['CODE_GENDER'].astype(str) + '_' + combined['DAYS_EMPLOYED'].astype(str)

k1_counts = combined['client_key_1'].map(combined['client_key_1'].value_counts())
k2_counts = combined['client_key_2'].map(combined['client_key_2'].value_counts())

train_df['client_app_count_1'] = k1_counts.iloc[:N_tr].values
test_df['client_app_count_1'] = k1_counts.iloc[N_tr:].values
train_df['client_app_count_2'] = k2_counts.iloc[:N_tr].values
test_df['client_app_count_2'] = k2_counts.iloc[N_tr:].values
train_df['is_repeat_client'] = ((k1_counts.iloc[:N_tr] > 1) | (k2_counts.iloc[:N_tr] > 1)).astype(int).values
test_df['is_repeat_client'] = ((k1_counts.iloc[N_tr:] > 1) | (k2_counts.iloc[N_tr:] > 1)).astype(int).values

train_term_raw = (train_df['AMT_CREDIT'] / (train_df['AMT_ANNUITY'] + 1.0)).values.reshape(-1, 1)
km_term = KMeans(n_clusters=8, random_state=42, n_init=10).fit(train_term_raw)
term_centers = np.sort(km_term.cluster_centers_.flatten())

def build_features(df):
    data = df.copy()
    data['is_pensioner_anomaly'] = (data['DAYS_EMPLOYED'] == 365243).astype(int)
    data['DAYS_EMPLOYED_clean'] = data['DAYS_EMPLOYED'].replace({365243: np.nan})

    data['age_years'] = -data['DAYS_BIRTH'] / 365.25
    data['employed_years'] = -data['DAYS_EMPLOYED_clean'] / 365.25
    data['career_retention_ratio'] = data['DAYS_EMPLOYED_clean'] / (data['DAYS_BIRTH'] - 18.0 * 365.25)

    data['root_credit'] = np.sqrt(np.maximum(0, data['AMT_CREDIT']))
    data['root_annuity'] = np.sqrt(np.maximum(0, data['AMT_ANNUITY']))
    data['root_goods'] = np.sqrt(np.maximum(0, data['AMT_GOODS_PRICE']))
    data['root_income'] = np.sqrt(np.maximum(0, data['AMT_INCOME_TOTAL']))
    data['root_age'] = np.sqrt(np.maximum(0, data['age_years']))
    data['root_employed'] = np.sqrt(np.maximum(0, data['employed_years']))

    data['payment_rate'] = data['AMT_ANNUITY'] / (data['AMT_CREDIT'] + 1.0)
    data['credit_to_income'] = data['AMT_CREDIT'] / (data['AMT_INCOME_TOTAL'] + 1.0)
    data['annuity_to_income'] = data['AMT_ANNUITY'] / (data['AMT_INCOME_TOTAL'] + 1.0)
    data['income_per_fam'] = data['AMT_INCOME_TOTAL'] / (data['CNT_FAM_MEMBERS'] + 1.0)
    data['goods_to_credit'] = data['AMT_GOODS_PRICE'] / (data['AMT_CREDIT'] + 1.0)
    data['credit_minus_goods'] = data['AMT_CREDIT'] - data['AMT_GOODS_PRICE']
    data['monthly_annuity_burden'] = data['AMT_ANNUITY'] - (data['AMT_INCOME_TOTAL'] / 12.0)
    data['annuity_to_goods'] = data['AMT_ANNUITY'] / (data['AMT_GOODS_PRICE'] + 1.0)

    data['ltv_ratio'] = data['AMT_CREDIT'] / (data['AMT_GOODS_PRICE'] + 1.0)
    data['is_overfinanced'] = (data['AMT_CREDIT'] > data['AMT_GOODS_PRICE']).astype(int)
    data['downpayment'] = (data['AMT_GOODS_PRICE'] - data['AMT_CREDIT']).clip(lower=0)
    data['downpayment_rate'] = data['downpayment'] / (data['AMT_GOODS_PRICE'] + 1.0)

    data['merton_solvency_buffer'] = (data['AMT_INCOME_TOTAL'] - 12.0 * data['AMT_ANNUITY']) / (data['AMT_CREDIT'] + 1.0)
    data['goods_markup_ratio'] = (data['AMT_CREDIT'] - data['AMT_GOODS_PRICE']) / (data['AMT_GOODS_PRICE'] + 1.0)

    data['cnt_adults'] = (data['CNT_FAM_MEMBERS'] - data['CNT_CHILDREN']).clip(lower=1.0)
    data['income_per_adult'] = data['AMT_INCOME_TOTAL'] / data['cnt_adults']
    data['children_ratio'] = data['CNT_CHILDREN'] / (data['CNT_FAM_MEMBERS'] + 1.0)

    term = (data['AMT_CREDIT'] / (data['AMT_ANNUITY'] + 1.0)).values.reshape(-1, 1)
    diffs = term - term_centers
    nearest_idx = np.argmin(np.abs(diffs), axis=1)
    nearest_center = term_centers[nearest_idx]

    data['term_raw'] = term.flatten()
    data['term_round'] = np.round(term.flatten()).astype(int)
    data['nearest_segment_peak'] = nearest_center
    data['dist_to_segment_peak'] = term.flatten() - nearest_center
    data['abs_dist_to_segment_peak'] = np.abs(term.flatten() - nearest_center)
    data['segment_cluster_id'] = nearest_idx
    data['implied_interest_charge'] = (data['AMT_ANNUITY'] * nearest_center - data['AMT_CREDIT']) / (data['AMT_CREDIT'] + 1.0)
    data['effective_interest_rate'] = 24.0 * (data['AMT_ANNUITY'] * data['term_raw'] - data['AMT_CREDIT']) / (data['AMT_CREDIT'] * (data['term_raw'] + 1.0) + 1.0)

    credit_bin = (np.round(data['AMT_CREDIT'] / 50000.0) * 50).astype(int)
    data['product_group'] = credit_bin.astype(str) + '_' + data['term_round'].astype(str)

    data['employed_to_birth'] = data['DAYS_EMPLOYED_clean'] / (data['DAYS_BIRTH'] - 1.0)
    data['car_to_birth'] = data['OWN_CAR_AGE'] / (data['age_years'] + 1.0)
    data['car_to_employed'] = data['OWN_CAR_AGE'] / (data['employed_years'] + 1.0)
    data['phone_to_birth'] = data['DAYS_LAST_PHONE_CHANGE'] / (data['DAYS_BIRTH'] - 1.0)
    data['phone_to_employed'] = data['DAYS_LAST_PHONE_CHANGE'] / (data['DAYS_EMPLOYED_clean'] - 1.0)
    data['id_to_birth'] = data['DAYS_ID_PUBLISH'] / (data['DAYS_BIRTH'] - 1.0)
    data['reg_to_birth'] = data['DAYS_REGISTRATION'] / (data['DAYS_BIRTH'] - 1.0)

    data['ext_1_hybrid'] = data['EXT_SOURCE_1'].fillna(data['EXT_SOURCE_1_PRED'])
    data['ext_3_hybrid'] = data['EXT_SOURCE_3'].fillna(data['EXT_SOURCE_3_PRED'])

    eps = 1e-4
    e2_filled = data['EXT_SOURCE_2'].fillna(0.5)
    l1 = np.log(data['ext_1_hybrid'].clip(eps, 1.0 - eps) / (1.0 - data['ext_1_hybrid'].clip(eps, 1.0 - eps)))
    l2 = np.log(e2_filled.clip(eps, 1.0 - eps) / (1.0 - e2_filled.clip(eps, 1.0 - eps)))
    l3 = np.log(data['ext_3_hybrid'].clip(eps, 1.0 - eps) / (1.0 - data['ext_3_hybrid'].clip(eps, 1.0 - eps)))
    data['logit_ext_mean'] = (l1 + l2 + l3) / 3.0
    data['log_odds_ext_consensus'] = 0.35 * l1 + 0.35 * l2 + 0.30 * l3

    data['ext_1_x_ext_2'] = data['ext_1_hybrid'] * data['EXT_SOURCE_2']
    data['ext_2_x_ext_3'] = data['EXT_SOURCE_2'] * data['ext_3_hybrid']
    data['ext_1_x_ext_3'] = data['ext_1_hybrid'] * data['ext_3_hybrid']

    bureau_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    data['bureau_enquiries_total'] = data[bureau_cols].sum(axis=1)
    return data

train_fe = build_features(train_df)
test_fe = build_features(test_df)

full_data = pd.concat([train_fe, test_fe], ignore_index=True)
for group_col in ['OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE']:
    for num_col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'EXT_SOURCE_2']:
        grp = full_data.groupby(group_col)[num_col].agg(['mean', 'median'])
        full_data[f'{num_col}_by_{group_col}_mean'] = full_data[group_col].map(grp['mean'])
        full_data[f'{num_col}_by_{group_col}_diff'] = full_data[num_col] - full_data[f'{num_col}_by_{group_col}_mean']

for cat in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE', 'REGION_POPULATION_RELATIVE', 'segment_cluster_id']:
    freq = full_data[cat].value_counts()
    full_data[f'{cat}_freq'] = full_data[cat].map(freq)

full_data['org_occ'] = full_data['ORGANIZATION_TYPE'].astype(str) + '_' + full_data['OCCUPATION_TYPE'].astype(str)
full_data['edu_gender'] = full_data['NAME_EDUCATION_TYPE'].astype(str) + '_' + full_data['CODE_GENDER'].astype(str)

core_pkls = [
    "accepted_subbin_features_full.pkl",
    "accepted_actuarial_features_full.pkl",
    "accepted_v5_discrete_tariffs_full.pkl",
    "accepted_v6_osw_bipartite_full.pkl"
]
core_loaded = []
for pkl_name in core_pkls:
    p_path = core_feat_dir / pkl_name
    if p_path.exists():
        cdf = pd.read_pickle(p_path)
        for c in cdf.columns:
            full_data[c] = cdf[c].values
            core_loaded.append(c)

new_features_loaded = []
if accepted_dir.exists():
    parquet_files = sorted(list(accepted_dir.glob("*.parquet")))
    for pf in parquet_files:
        try:
            feat_df = pd.read_parquet(pf)
            for c in feat_df.columns:
                if c != 'SK_ID_CURR' and c not in full_data.columns:
                    full_data[c] = feat_df[c].values
                    new_features_loaded.append(c)
        except Exception as e:
            print(f"Warning: Failed loading {pf.name}: {e}")

print(f"Loaded {len(core_loaded)} core v2-v6 features + {len(new_features_loaded)} newly discovered features: {new_features_loaded}", flush=True)

has_tariff_col = 'cand_tariff_joint_ext_lo_ortho' in full_data.columns
subspace_cols = [
    'ext_1_hybrid', 'EXT_SOURCE_2', 'ext_3_hybrid', 'logit_ext_mean',
    'payment_rate', 'annuity_to_income', 'credit_to_income', 'goods_to_credit',
    'merton_solvency_buffer', 'age_years', 'employed_years'
] + (['cand_tariff_joint_ext_lo_ortho'] if has_tariff_col else [])

Z_sub_all = np.column_stack([to_probit(full_data[c].values) for c in subspace_cols])
z_ext_all = to_probit(full_data['logit_ext_mean'].values)
k_dim = Z_sub_all.shape[1]
tau_grid = np.linspace(1.0 / 32.0, 31.0 / 32.0, 31)
te_idx = np.arange(N_tr, N_all)

cat_cols = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'org_occ', 'edu_gender',
    'segment_cluster_id', 'product_group'
]
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
full_data[cat_cols] = encoder.fit_transform(full_data[cat_cols].astype(str))

train_base = full_data.iloc[:N_tr].copy()
test_base = full_data.iloc[N_tr:].copy()
y = train_base['TARGET'].values.astype(np.int32)
prior = y.mean()

ignore_cols = ['SK_ID_CURR', 'TARGET', 'client_key_1', 'client_key_2', 'DAYS_EMPLOYED_clean', 'days_employed_clean']
target_enc_cols = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'org_occ', 'edu_gender', 'term_round', 'product_group']

sphere_source_cols = [
    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'ext_1_hybrid', 'EXT_SOURCE_2', 'ext_3_hybrid', 'payment_rate', 'credit_to_income', 'merton_solvency_buffer'
]

seeds = [1111, 2222, 3333, 4444, 5555]
n_seeds = len(seeds)
n_splits = 10
models_to_train = ['lgb_cons', 'xgb_cuda', 'cb_gpu_lossguide']

oof_accum = {k: np.zeros(N_tr) for k in models_to_train}
test_accum = {k: np.zeros(N_te) for k in models_to_train}

save_npz = output_dir / "oof_and_test_cloud_burst_5seed.npz"

for s_idx, current_seed in enumerate(seeds):
    s_t0 = time.time()
    train_s = train_base.copy()
    test_s = test_base.copy()

    train_s['cos_directional_margin'] = np.nan
    train_s['cos_sim_safe'] = np.nan
    train_s['cos_sim_default'] = np.nan
    train_s['hypersphere_radius'] = np.nan

    test_s['cos_directional_margin'] = 0.0
    test_s['cos_sim_safe'] = 0.0
    test_s['cos_sim_default'] = 0.0
    test_s['hypersphere_radius'] = 0.0

    for col in target_enc_cols:
        train_s[f'{col}_te'] = np.nan
        test_s[f'{col}_te'] = np.zeros(N_te)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
    X_sph_tr_raw = train_s[sphere_source_cols].fillna(0).values
    X_sph_te_raw = test_s[sphere_source_cols].fillna(0).values

    V_ortho_seed = haar_orthogonal_directions(k_dim, n_blocks=4, seed=current_seed)
    P_lin_seed = Z_sub_all @ V_ortho_seed.T
    osw_w1_seed = np.zeros(N_all, dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_s, y)):
        scaler = StandardScaler().fit(X_sph_tr_raw[tr_idx])
        X_std_tr = scaler.transform(X_sph_tr_raw[tr_idx])
        X_std_val = scaler.transform(X_sph_tr_raw[val_idx])
        X_std_te = scaler.transform(X_sph_te_raw)

        norm_tr = np.linalg.norm(X_std_tr, axis=1, keepdims=True) + 1e-9
        norm_val = np.linalg.norm(X_std_val, axis=1, keepdims=True) + 1e-9
        norm_te = np.linalg.norm(X_std_te, axis=1, keepdims=True) + 1e-9

        Z_tr = X_std_tr / norm_tr
        Z_val = X_std_val / norm_val
        Z_te = X_std_te / norm_te

        y_tr_fold = y[tr_idx]
        c_safe = Z_tr[y_tr_fold == 0].mean(axis=0)
        c_safe /= (np.linalg.norm(c_safe) + 1e-9)

        c_def = Z_tr[y_tr_fold == 1].mean(axis=0)
        c_def /= (np.linalg.norm(c_def) + 1e-9)

        cos_s_val = np.dot(Z_val, c_safe)
        cos_d_val = np.dot(Z_val, c_def)
        train_s.iloc[val_idx, train_s.columns.get_loc('cos_sim_safe')] = cos_s_val
        train_s.iloc[val_idx, train_s.columns.get_loc('cos_sim_default')] = cos_d_val
        train_s.iloc[val_idx, train_s.columns.get_loc('cos_directional_margin')] = cos_s_val - cos_d_val
        train_s.iloc[val_idx, train_s.columns.get_loc('hypersphere_radius')] = norm_val.flatten()

        cos_s_te = np.dot(Z_te, c_safe)
        cos_d_te = np.dot(Z_te, c_def)
        test_s['cos_sim_safe'] += cos_s_te / float(n_splits)
        test_s['cos_sim_default'] += cos_d_te / float(n_splits)
        test_s['cos_directional_margin'] += (cos_s_te - cos_d_te) / float(n_splits)
        test_s['hypersphere_radius'] += norm_te.flatten() / float(n_splits)

        idx_s = tr_idx[y_tr_fold == 0]
        idx_d = tr_idx[y_tr_fold == 1]
        q_lin_s = np.quantile(P_lin_seed[idx_s], tau_grid, axis=0)
        q_lin_d = np.quantile(P_lin_seed[idx_d], tau_grid, axis=0)
        osw_w1_seed[val_idx] = mean_w1_to_quantiles(P_lin_seed[val_idx], q_lin_s) - mean_w1_to_quantiles(P_lin_seed[val_idx], q_lin_d)
        osw_w1_seed[te_idx] += (mean_w1_to_quantiles(P_lin_seed[te_idx], q_lin_s) - mean_w1_to_quantiles(P_lin_seed[te_idx], q_lin_d)) / float(n_splits)

    cand_osw_seed = to_probit_split(osw_w1_seed, N_tr) + 0.62 * z_ext_all
    train_s['cand_osw_w1_plus_ext_062'] = cand_osw_seed[:N_tr]
    test_s['cand_osw_w1_plus_ext_062'] = cand_osw_seed[N_tr:]

    for train_idx, val_idx in skf.split(train_s, y):
        tr = train_s.iloc[train_idx]
        val = train_s.iloc[val_idx]
        for col in target_enc_cols:
            counts = tr[col].value_counts()
            sums = tr.groupby(col)['TARGET'].sum()
            smoothed = (sums + 20.0 * prior) / (counts + 20.0)
            train_s.iloc[val_idx, train_s.columns.get_loc(f'{col}_te')] = val[col].map(smoothed).fillna(prior).values

    for col in target_enc_cols:
        counts = train_s[col].value_counts()
        sums = train_s.groupby(col)['TARGET'].sum()
        smoothed = (sums + 20.0 * prior) / (counts + 20.0)
        test_s[f'{col}_te'] = test_s[col].map(smoothed).fillna(prior).values

    features = [c for c in train_s.columns if c not in ignore_cols]
    X_s = train_s[features].copy()
    X_test_s = test_s[features].copy()
    if s_idx == 0:
        print(f"Cloud Burst Training ({len(features)} features) on GPU...", flush=True)

    xgb_cuda_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'device': 'cuda', 'learning_rate': 0.025, 'max_depth': 6,
        'subsample': 0.85, 'colsample_bytree': 0.40, 'reg_alpha': 0.5, 'reg_lambda': 25.0,
        'random_state': current_seed, 'n_estimators': 3500, 'early_stopping_rounds': 150
    }

    lgb_cons_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.020, 'num_leaves': 45, 'max_depth': 6,
        'min_child_samples': 65, 'subsample': 0.85, 'subsample_freq': 1,
        'colsample_bytree': 0.45, 'reg_alpha': 0.5, 'reg_lambda': 30.0,
        'random_state': current_seed, 'n_estimators': 3500, 'verbose': -1, 'n_jobs': -1
    }

    cb_lossguide_params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC', 'learning_rate': 0.025,
        'grow_policy': 'Lossguide', 'max_leaves': 48, 'min_data_in_leaf': 120,
        'l2_leaf_reg': 20.0, 'random_strength': 1.5, 'iterations': 3500,
        'task_type': 'GPU', 'random_seed': current_seed, 'verbose': False
    }

    seed_oof = {k: np.zeros(N_tr) for k in models_to_train}
    seed_test = {k: np.zeros(N_te) for k in models_to_train}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_s, y)):
        X_train, y_train = X_s.iloc[train_idx], y[train_idx]
        X_val, y_val = X_s.iloc[val_idx], y[val_idx]

        m_xgb = xgb.XGBClassifier(**xgb_cuda_params)
        m_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        seed_oof['xgb_cuda'][val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        seed_test['xgb_cuda'] += m_xgb.predict_proba(X_test_s)[:, 1] / float(n_splits)

        m_lgb = lgb.LGBMClassifier(**lgb_cons_params)
        m_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150, verbose=False)])
        seed_oof['lgb_cons'][val_idx] = m_lgb.predict_proba(X_val)[:, 1]
        seed_test['lgb_cons'] += m_lgb.predict_proba(X_test_s)[:, 1] / float(n_splits)

        m_cb = CatBoostClassifier(**cb_lossguide_params)
        m_cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=150, verbose=False)
        seed_oof['cb_gpu_lossguide'][val_idx] = m_cb.predict_proba(X_val)[:, 1]
        seed_test['cb_gpu_lossguide'] += m_cb.predict_proba(X_test_s)[:, 1] / float(n_splits)

        del m_xgb, m_lgb, m_cb
        gc.collect()

    for k in models_to_train:
        oof_accum[k] += seed_oof[k]
        test_accum[k] += seed_test[k]

    cur_count = float(s_idx + 1)
    oof_cur = {k: oof_accum[k] / cur_count for k in models_to_train}
    test_cur = {k: test_accum[k] / cur_count for k in models_to_train}

    auc_l = roc_auc_score(y, oof_cur['lgb_cons'])
    auc_x = roc_auc_score(y, oof_cur['xgb_cuda'])
    auc_c = roc_auc_score(y, oof_cur['cb_gpu_lossguide'])
    print(f"Seed {s_idx+1}/{n_seeds} Done ({time.time()-s_t0:.1f}s) | Running OOF: LGB={auc_l:.6f} | XGB={auc_x:.6f} | CB_LG={auc_c:.6f}", flush=True)

    np.savez_compressed(
        save_npz,
        y_true=y,
        oof_lgb_cons=oof_cur['lgb_cons'],
        test_lgb_cons=test_cur['lgb_cons'],
        oof_xgb_cuda=oof_cur['xgb_cuda'],
        test_xgb_cuda=test_cur['xgb_cuda'],
        oof_cb_lossguide=oof_cur['cb_gpu_lossguide'],
        test_cb_lossguide=test_cur['cb_gpu_lossguide'],
        completed_seeds=cur_count
    )

print(f"\nAll {n_seeds} Burst Seeds Complete in {time.time()-start_time:.1f}s! Saved to {save_npz}")
