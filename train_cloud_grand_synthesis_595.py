import os
import sys
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, norm

from train_cloud_dae_tabm463 import (
    load_data,
    to_average_uniform_rank_f64,
    to_strict_uniform_rank_f64
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

def is_prebaked_target_encoded_column(col_name: str) -> bool:
    u = str(col_name).upper()
    if u.endswith("_TE") or "_TE_" in u or u.startswith("BUH_TE_"):
        return True
    if "BUHLMANN_" in u or "UIDY_" in u or "UID2Y_" in u:
        return True
    return False

def select_non_overlapping_tier2_columns(
    X_t4_tr: np.ndarray,
    X_t2_tr: np.ndarray,
    y_tr: np.ndarray,
    t4_names: np.ndarray,
    t2_names: np.ndarray,
    corr_threshold_t4: float = 0.960,
    corr_threshold_internal: float = 0.950
):
    t4_name_set = set(str(c) for c in t4_names)
    n_sub = min(35000, X_t4_tr.shape[0])
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(X_t4_tr.shape[0], size=n_sub, replace=False)
    A_raw = X_t4_tr[sub_idx].astype(np.float64)
    B_raw = X_t2_tr[sub_idx].astype(np.float64)
    y_sub = y_tr[sub_idx].astype(np.float64)
    y_norm = (y_sub - y_sub.mean()) / (y_sub.std() + 1e-12)
    A_med = np.where(np.isnan(np.nanmedian(A_raw, axis=0, keepdims=True)), 0.0, np.nanmedian(A_raw, axis=0, keepdims=True))
    B_med = np.where(np.isnan(np.nanmedian(B_raw, axis=0, keepdims=True)), 0.0, np.nanmedian(B_raw, axis=0, keepdims=True))
    A = np.where(np.isnan(A_raw), A_med, A_raw)
    B = np.where(np.isnan(B_raw), B_med, B_raw)
    A_std = A.std(axis=0, keepdims=True)
    B_std = B.std(axis=0, keepdims=True)
    A_norm = (A - A.mean(axis=0, keepdims=True)) / np.where(A_std < 1e-12, 1.0, A_std)
    B_norm = (B - B.mean(axis=0, keepdims=True)) / np.where(B_std < 1e-12, 1.0, B_std)
    corr_mat_t4 = np.abs((B_norm.T @ A_norm) / float(n_sub))
    max_corr_vs_t4 = corr_mat_t4.max(axis=1)
    corr_mat_b = np.abs((B_norm.T @ B_norm) / float(n_sub))
    target_relevance = np.abs((B_norm.T @ y_norm) / float(n_sub))
    cand_order = np.argsort(-target_relevance)
    keep_indices = []
    for j in cand_order:
        j = int(j)
        col_name = str(t2_names[j])
        if col_name in t4_name_set:
            continue
        if is_prebaked_target_encoded_column(col_name):
            continue
        if B_std[0, j] <= 1e-9 or max_corr_vs_t4[j] >= corr_threshold_t4:
            continue
        if any(corr_mat_b[j, k] >= corr_threshold_internal for k in keep_indices):
            continue
        keep_indices.append(j)
    keep_indices = np.sort(np.array(keep_indices, dtype=np.int64))
    return keep_indices, [f"T2_ORTH_{t2_names[j]}" for j in keep_indices]

def precompute_tariff_and_entity_structures(X_t2_tr: np.ndarray, X_t2_te: np.ndarray, t2_names: np.ndarray):
    name_to_idx = {str(c): idx for idx, c in enumerate(t2_names)}
    X_t2_all = np.vstack([X_t2_tr, X_t2_te])
    n_all = X_t2_all.shape[0]
    cat_cols_list = []
    cat_sizes = []
    cat_names = []
    for c_name in t2_names:
        s_name = str(c_name)
        if s_name.startswith("buh_CAT_"):
            raw_vals = np.nan_to_num(X_t2_all[:, name_to_idx[s_name]], nan=-1.0).round(0).astype(np.int64)
            codes, uniques = pd.factorize(raw_vals)
            cat_cols_list.append(codes.astype(np.int32))
            cat_sizes.append(len(uniques))
            cat_names.append(f"FOLD_BUH_{s_name.replace('buh_CAT_', '')}")
    cred = np.nan_to_num(X_t2_all[:, name_to_idx["AMT_CREDIT"]].astype(np.float64), nan=0.0)
    ann = np.nan_to_num(X_t2_all[:, name_to_idx["AMT_ANNUITY"]].astype(np.float64), nan=0.0)
    goods = np.nan_to_num(X_t2_all[:, name_to_idx["AMT_GOODS_PRICE"]].astype(np.float64), nan=0.0)
    h_ratio3 = np.round(ann / np.maximum(cred, 1.0), 3)
    h_ratio4 = np.round(ann / np.maximum(cred, 1.0), 4)
    h_ratio5 = np.round(ann / np.maximum(cred, 1.0), 5)
    h_ag3 = np.round(ann / np.maximum(goods, 1.0), 3)
    h_ag4 = np.round(ann / np.maximum(goods, 1.0), 4)
    h_cg3 = np.round(cred / np.maximum(goods, 1.0), 3)
    h_eq = (np.abs(cred - goods) < 0.5).astype(int)
    h3_r4_eq = pd.Series(h_ratio4).astype(str) + "_" + pd.Series(h_eq).astype(str)
    extra_tariffs = [
        ("FOLD_BUH_H_RATIO3", h_ratio3),
        ("FOLD_BUH_H_RATIO4", h_ratio4),
        ("FOLD_BUH_H_RATIO5", h_ratio5),
        ("FOLD_BUH_H_AG3", h_ag3),
        ("FOLD_BUH_H_AG4", h_ag4),
        ("FOLD_BUH_H_CG3", h_cg3),
        ("FOLD_BUH_H3_R4_EQ", h3_r4_eq)
    ]
    for t_name, t_arr in extra_tariffs:
        codes, uniques = pd.factorize(t_arr)
        cat_cols_list.append(codes.astype(np.int32))
        cat_sizes.append(len(uniques))
        cat_names.append(t_name)
    code_matrix = np.column_stack(cat_cols_list).astype(np.int32)
    d_birth = np.nan_to_num(X_t2_all[:, name_to_idx["DAYS_BIRTH"]].astype(np.float64), nan=0.0).round(0)
    d_reg = np.nan_to_num(X_t2_all[:, name_to_idx["DAYS_REGISTRATION"]].astype(np.float64), nan=0.0).round(0)
    d_id = np.nan_to_num(X_t2_all[:, name_to_idx["DAYS_ID_PUBLISH"]].astype(np.float64), nan=0.0).round(0)
    d_emp = np.nan_to_num(X_t2_all[:, name_to_idx["DAYS_EMPLOYED"]].astype(np.float64), nan=365243.0).round(0)
    d_ph = np.nan_to_num(X_t2_all[:, name_to_idx["DAYS_LAST_PHONE_CHANGE"]].astype(np.float64), nan=0.0).round(0)
    gender = np.nan_to_num(X_t2_all[:, name_to_idx["CODE_GENDER"]].astype(np.float64), nan=-1.0).round(0).astype(int)
    edu = np.nan_to_num(X_t2_all[:, name_to_idx["NAME_EDUCATION_TYPE"]].astype(np.float64), nan=-1.0).round(0).astype(int)
    df_ent = pd.DataFrame({
        "row": np.arange(n_all, dtype=np.int32),
        "DAYS_BIRTH": d_birth,
        "DAYS_EMPLOYED": d_emp,
        "DAYS_LAST_PHONE_CHANGE": d_ph,
        "CODE_GENDER": gender,
        "NAME_EDUCATION_TYPE": edu
    })
    df_ent["k2"] = (
        pd.Series(d_birth - d_reg).astype(int).astype(str) + "|" +
        pd.Series(d_birth - d_id).astype(int).astype(str) + "|" +
        pd.Series(gender).astype(str) + "|" +
        pd.Series(edu).astype(str)
    )
    vc = df_ent["k2"].value_counts()
    G2 = df_ent[df_ent["k2"].map(vc) >= 2]
    P2 = G2.merge(G2, on="k2", suffixes=("", "_p"))
    P2 = P2[P2["row"] != P2["row_p"]].copy()
    diff_b = P2["DAYS_BIRTH"] - P2["DAYS_BIRTH_p"]
    emp_ok = (
        ((P2["DAYS_EMPLOYED"] == 365243) & (P2["DAYS_EMPLOYED_p"] == 365243)) |
        ((P2["DAYS_EMPLOYED"] - P2["DAYS_EMPLOYED_p"] == diff_b) & (P2["DAYS_EMPLOYED"] != 365243) & (P2["DAYS_EMPLOYED_p"] != 365243))
    )
    ph_ok = (
        (P2["DAYS_LAST_PHONE_CHANGE"] - P2["DAYS_LAST_PHONE_CHANGE_p"] == diff_b) |
        (P2["DAYS_LAST_PHONE_CHANGE"] == P2["DAYS_LAST_PHONE_CHANGE_p"])
    )
    P2["d"] = diff_b
    P2["ad"] = diff_b.abs()
    P2["tier"] = np.where((diff_b == 0) | (emp_ok & ph_ok), 2, np.where(emp_ok | ph_ok, 1, 0))
    confirmed_pairs = P2[P2["tier"] == 2][["row", "row_p", "d", "ad"]].reset_index(drop=True)
    soft_t1_pairs = P2[P2["tier"] == 1][["row", "row_p", "d", "ad"]].reset_index(drop=True)
    dyn_names = cat_names + [
        "FOLD_UIDY_N", "FOLD_UIDY_SUM", "FOLD_UIDY_NEAR_Y", "FOLD_UIDY_NEAR_D", "FOLD_UIDY_ANYBAD",
        "FOLD_UID2Y_T1_N", "FOLD_UID2Y_T1_SUM", "FOLD_UID2Y_T1_NEAR_Y", "FOLD_UID2Y_T1_NEAR_D", "FOLD_UID2Y_T1_ANYBAD"
    ]
    return code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, dyn_names, n_all

def calc_buhlmann_vector(c_codes: np.ndarray, yy: np.ndarray, n_sz: int) -> np.ndarray:
    counts = np.bincount(c_codes, minlength=n_sz).astype(np.float64)
    sums = np.bincount(c_codes, weights=yy, minlength=n_sz).astype(np.float64)
    mask = counts > 0
    y_c = np.zeros(n_sz, dtype=np.float64)
    y_c[mask] = sums[mask] / counts[mask]
    y_bar = float(np.mean(yy))
    n_total = float(len(yy))
    c_active = float(np.sum(mask))
    epv = np.sum(counts[mask] * y_c[mask] * (1.0 - y_c[mask])) / max(n_total - c_active, 1.0)
    epv = max(epv, 1e-5)
    ss_between = np.sum(counts[mask] * (y_c[mask] - y_bar) ** 2)
    c_eff = n_total - np.sum(counts[mask] ** 2) / n_total
    vhm = (ss_between - (c_active - 1.0) * epv) / max(c_eff, 1e-5)
    vhm = max(vhm, 1e-6)
    m_adaptive = np.clip(epv / vhm, 5.0, 300.0)
    z_c = counts / (counts + m_adaptive)
    encoded = z_c * y_c + (1.0 - z_c) * y_bar
    return encoded.astype(np.float32)

def compute_uid_target_block(rows: np.ndarray, lab: np.ndarray, target_pairs: pd.DataFrame) -> np.ndarray:
    inr = np.zeros(len(lab), dtype=bool)
    inr[rows] = True
    r_arr = target_pairs["row"].to_numpy()
    rp_arr = target_pairs["row_p"].to_numpy()
    m = inr[r_arr] & ~np.isnan(lab[rp_arr])
    Q = target_pairs[m].assign(yp=lab[rp_arr[m]])
    g = Q.groupby("row")
    near = Q.sort_values("ad").groupby("row").first()
    F = pd.DataFrame({
        "N": g.size(),
        "SUM": g["yp"].sum(),
        "NEAR_Y": near["yp"],
        "NEAR_D": near["ad"]
    }).reindex(rows)
    F["ANYBAD"] = np.where(F["N"].isna(), np.nan, (F["SUM"] > 0).astype(np.float32))
    return F.to_numpy(dtype=np.float32)

def compute_fold_isolated_dynamic_features(
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    y_tr_full: np.ndarray,
    n_train: int,
    n_all: int,
    code_matrix: np.ndarray,
    cat_sizes: list,
    confirmed_pairs: pd.DataFrame,
    soft_t1_pairs: pd.DataFrame,
    fold_seed: int
):
    n_cats = len(cat_sizes)
    tr_buh = np.zeros((len(tr_idx), n_cats), dtype=np.float32)
    va_buh = np.zeros((len(va_idx), n_cats), dtype=np.float32)
    te_buh = np.zeros((n_all - n_train, n_cats), dtype=np.float32)
    y_f_tr = y_tr_full[tr_idx]
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for in_tr, in_va in inner_skf.split(np.zeros(len(tr_idx)), y_f_tr):
        abs_in_tr = tr_idx[in_tr]
        abs_in_va = tr_idx[in_va]
        y_in = y_tr_full[abs_in_tr]
        for col_j, n_sz in enumerate(cat_sizes):
            enc_vec = calc_buhlmann_vector(code_matrix[abs_in_tr, col_j], y_in, n_sz)
            tr_buh[in_va, col_j] = enc_vec[code_matrix[abs_in_va, col_j]]
    for col_j, n_sz in enumerate(cat_sizes):
        enc_full_tr = calc_buhlmann_vector(code_matrix[tr_idx, col_j], y_f_tr, n_sz)
        va_buh[:, col_j] = enc_full_tr[code_matrix[va_idx, col_j]]
        te_buh[:, col_j] = enc_full_tr[code_matrix[n_train:, col_j]]
    lab_fold = np.full(n_all, np.nan, dtype=np.float64)
    lab_fold[tr_idx] = y_f_tr.astype(np.float64)
    te_idx = np.arange(n_train, n_all, dtype=np.int64)
    uidy_tr = compute_uid_target_block(tr_idx, lab_fold, confirmed_pairs)
    uidy_va = compute_uid_target_block(va_idx, lab_fold, confirmed_pairs)
    uidy_te = compute_uid_target_block(te_idx, lab_fold, confirmed_pairs)
    uid2y_tr = compute_uid_target_block(tr_idx, lab_fold, soft_t1_pairs)
    uid2y_va = compute_uid_target_block(va_idx, lab_fold, soft_t1_pairs)
    uid2y_te = compute_uid_target_block(te_idx, lab_fold, soft_t1_pairs)
    dyn_tr = np.hstack([tr_buh, uidy_tr, uid2y_tr]).astype(np.float32)
    dyn_va = np.hstack([va_buh, uidy_va, uid2y_va]).astype(np.float32)
    dyn_te = np.hstack([te_buh, uidy_te, uid2y_te]).astype(np.float32)
    return dyn_tr, dyn_va, dyn_te

def build_grand_synthesis_matrix():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_t4_tr, X_t4_te, y_tr, sk_id_test, t4_cols = load_data()
    t2_npz_cands = [
        "data/matrix_226_features_buhlmann_tier2.npz",
        "matrix_226_features_buhlmann_tier2.npz",
        "output/matrix_226_features_buhlmann_tier2.npz",
        os.path.join(script_dir, "data", "matrix_226_features_buhlmann_tier2.npz"),
        os.path.join(script_dir, "..", "output", "matrix_226_features_buhlmann_tier2.npz")
    ]
    t2_path = next((p for p in t2_npz_cands if os.path.exists(p)), None)
    if t2_path is None:
        shard_dirs = [os.path.join(script_dir, "data"), "data", script_dir, "."]
        for sdir in shard_dirs:
            shards = sorted(glob.glob(os.path.join(sdir, "matrix_226_features_buhlmann_tier2_part_*.bin")))
            if len(shards) > 0:
                reassembled_path = os.path.join(sdir, "matrix_226_features_buhlmann_tier2.npz")
                with open(reassembled_path, "wb") as fout:
                    for sp in shards:
                        with open(sp, "rb") as fin:
                            fout.write(fin.read())
                t2_path = reassembled_path
                break
    if t2_path is None:
        raise FileNotFoundError("Missing matrix_226_features_buhlmann_tier2.npz or its binary shards.")
    t2_data = np.load(t2_path, allow_pickle=True)
    if "y_tr" in t2_data.files:
        assert np.array_equal(t2_data["y_tr"], y_tr), "Target label mismatch between Tier-4 and Tier-2 NPZ matrices."
    if "sk_id_curr_test" in t2_data.files:
        assert np.array_equal(t2_data["sk_id_curr_test"], sk_id_test), "Test SK_ID_CURR mismatch between Tier-4 and Tier-2."
    X_t2_tr = np.asarray(t2_data["X_tr"], dtype=np.float32)
    X_t2_te = np.asarray(t2_data["X_te"], dtype=np.float32)
    t2_names = t2_data["feature_names"] if "feature_names" in t2_data.files else np.arange(X_t2_tr.shape[1])
    X_t2_tr = np.where(np.isinf(X_t2_tr), np.nan, X_t2_tr)
    X_t2_te = np.where(np.isinf(X_t2_te), np.nan, X_t2_te)
    keep_idx, orth_t2_names = select_non_overlapping_tier2_columns(
        X_t4_tr, X_t2_tr, y_tr, t4_cols, t2_names, corr_threshold_t4=0.960, corr_threshold_internal=0.950
    )
    X_orth_tr = X_t2_tr[:, keep_idx]
    X_orth_te = X_t2_te[:, keep_idx]
    if X_orth_tr.shape[1] == 0:
        raise ValueError("Zero orthogonal Tier-2 columns selected.")
    code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, dyn_names, n_all = precompute_tariff_and_entity_structures(
        X_t2_tr, X_t2_te, t2_names
    )
    dae_cands = [
        "output/dae_128_latent_features.npz",
        "dae_128_latent_features.npz",
        "data/dae_128_latent_features.npz",
        os.path.join(script_dir, "output", "dae_128_latent_features.npz"),
        os.path.join(script_dir, "..", "output", "dae_128_latent_features.npz"),
        os.path.join(script_dir, "data", "dae_128_latent_features.npz")
    ]
    dae_path = next((p for p in dae_cands if os.path.exists(p)), None)
    if dae_path is None:
        raise FileNotFoundError("Missing dae_128_latent_features.npz. Run train_cloud_dae_tabm463.py first.")
    dd = np.load(dae_path, allow_pickle=True)
    dae_tr = np.asarray(dd["dae_tr"], dtype=np.float32)
    dae_te = np.asarray(dd["dae_te"], dtype=np.float32)
    svd = TruncatedSVD(n_components=24, random_state=42)
    svd_all = svd.fit_transform(np.vstack([dae_tr[:, :128], dae_te[:, :128]])).astype(np.float32)
    svd_tr = svd_all[:len(X_t4_tr)]
    svd_te = svd_all[len(X_t4_tr):]
    anom_tr = dae_tr[:, 128:]
    anom_te = dae_te[:, 128:]
    X_dae_tr = np.hstack([svd_tr, anom_tr]).astype(np.float32)
    X_dae_te = np.hstack([svd_te, anom_te]).astype(np.float32)
    dae_names = [f"DAE_SVD_{i:02d}" for i in range(24)] + [f"DAE_ANOM_{i:02d}" for i in range(anom_tr.shape[1])]
    X_base_tr = np.hstack([X_t4_tr, X_orth_tr, X_dae_tr]).astype(np.float32)
    X_base_te = np.hstack([X_t4_te, X_orth_te, X_dae_te]).astype(np.float32)
    all_feature_names = np.array(list(t4_cols) + orth_t2_names + dae_names + dyn_names)
    total_dim = X_base_tr.shape[1] + len(dyn_names)
    print("=" * 84, flush=True)
    print(f"BUILT GRAND SYNTHESIS EARLY-FUSION MATRIX: Train=({len(X_base_tr)}, {total_dim}), Test=({len(X_base_te)}, {total_dim})", flush=True)
    print(f"  -> Tier-4 Relational Features               : {X_t4_tr.shape[1]}", flush=True)
    print(f"  -> Deduplicated Target-Free Tier-2 Features : {X_orth_tr.shape[1]}", flush=True)
    print(f"  -> SVD-DAE & Anomaly Features               : {X_dae_tr.shape[1]}", flush=True)
    print(f"  -> Fold-Isolated Dynamic Buhlmann & UIDTE   : {len(dyn_names)}", flush=True)
    print("=" * 84, flush=True)
    dyn_pack = (code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, len(X_base_tr), n_all)
    return X_base_tr, X_base_te, y_tr, sk_id_test, all_feature_names, dyn_pack

def main():
    os.makedirs("output", exist_ok=True)
    output_npz_path = "output/oof_and_test_grand_synthesis_595.npz"
    X_base_tr, X_base_te, y_tr, sk_id_test, feat_names, dyn_pack = build_grand_synthesis_matrix()
    code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, n_train, n_all = dyn_pack
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_lgb_cons = np.zeros(len(X_base_tr), dtype=np.float64)
    tst_lgb_cons = np.zeros(len(X_base_te), dtype=np.float64)
    tst_lgb_cons_prob = np.zeros(len(X_base_te), dtype=np.float64)
    oof_lgb_deep = np.zeros(len(X_base_tr), dtype=np.float64)
    tst_lgb_deep = np.zeros(len(X_base_te), dtype=np.float64)
    tst_lgb_deep_prob = np.zeros(len(X_base_te), dtype=np.float64)
    oof_xgb_d6 = np.zeros(len(X_base_tr), dtype=np.float64)
    tst_xgb_d6 = np.zeros(len(X_base_te), dtype=np.float64)
    tst_xgb_d6_prob = np.zeros(len(X_base_te), dtype=np.float64)
    oof_cb_sym = np.zeros(len(X_base_tr), dtype=np.float64)
    tst_cb_sym = np.zeros(len(X_base_te), dtype=np.float64)
    tst_cb_sym_prob = np.zeros(len(X_base_te), dtype=np.float64)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_base_tr, y_tr), start=1):
        dyn_tr, dyn_va, dyn_te = compute_fold_isolated_dynamic_features(
            tr_idx=tr_idx,
            va_idx=va_idx,
            y_tr_full=y_tr,
            n_train=n_train,
            n_all=n_all,
            code_matrix=code_matrix,
            cat_sizes=cat_sizes,
            confirmed_pairs=confirmed_pairs,
            soft_t1_pairs=soft_t1_pairs,
            fold_seed=1000 + fold_idx
        )
        X_tr_f = np.hstack([X_base_tr[tr_idx], dyn_tr]).astype(np.float32)
        X_va_f = np.hstack([X_base_tr[va_idx], dyn_va]).astype(np.float32)
        X_te_f = np.hstack([X_base_te, dyn_te]).astype(np.float32)
        y_tr_f = y_tr[tr_idx]
        y_va_f = y_tr[va_idx]
        model_cons = lgb.LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.016,
            num_leaves=48,
            max_depth=6,
            min_child_samples=70,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.32,
            reg_alpha=10.0,
            reg_lambda=20.0,
            random_state=100 + fold_idx,
            n_jobs=-1,
            verbose=-1
        )
        model_cons.fit(
            X_tr_f, y_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            callbacks=[lgb.early_stopping(90, verbose=False)]
        )
        p_va_cons = model_cons.predict_proba(X_va_f)[:, 1]
        p_te_cons = model_cons.predict_proba(X_te_f)[:, 1]
        oof_lgb_cons[va_idx] = to_average_uniform_rank_f64(p_va_cons)
        tst_lgb_cons += to_average_uniform_rank_f64(p_te_cons) / n_splits
        tst_lgb_cons_prob += p_te_cons.astype(np.float64) / n_splits
        model_deep = lgb.LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.014,
            num_leaves=96,
            max_depth=7,
            min_child_samples=90,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.25,
            extra_trees=True,
            reg_alpha=15.0,
            reg_lambda=30.0,
            random_state=200 + fold_idx,
            n_jobs=-1,
            verbose=-1
        )
        model_deep.fit(
            X_tr_f, y_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            callbacks=[lgb.early_stopping(90, verbose=False)]
        )
        p_va_deep = model_deep.predict_proba(X_va_f)[:, 1]
        p_te_deep = model_deep.predict_proba(X_te_f)[:, 1]
        oof_lgb_deep[va_idx] = to_average_uniform_rank_f64(p_va_deep)
        tst_lgb_deep += to_average_uniform_rank_f64(p_te_deep) / n_splits
        tst_lgb_deep_prob += p_te_deep.astype(np.float64) / n_splits
        if HAS_XGBOOST:
            xgb_device = "cuda" if HAS_CUDA else "cpu"
            model_xgb = xgb.XGBClassifier(
                n_estimators=4000,
                learning_rate=0.018,
                max_depth=6,
                min_child_weight=50,
                subsample=0.80,
                colsample_bytree=0.28,
                reg_alpha=10.0,
                reg_lambda=25.0,
                tree_method="hist",
                device=xgb_device,
                eval_metric="auc",
                early_stopping_rounds=90,
                random_state=300 + fold_idx,
                n_jobs=-1,
                verbosity=0
            )
            model_xgb.fit(
                X_tr_f, y_tr_f,
                eval_set=[(X_va_f, y_va_f)],
                verbose=False
            )
            p_va_xgb = model_xgb.predict_proba(X_va_f)[:, 1]
            p_te_xgb = model_xgb.predict_proba(X_te_f)[:, 1]
            oof_xgb_d6[va_idx] = to_average_uniform_rank_f64(p_va_xgb)
            tst_xgb_d6 += to_average_uniform_rank_f64(p_te_xgb) / n_splits
            tst_xgb_d6_prob += p_te_xgb.astype(np.float64) / n_splits
        if HAS_CATBOOST:
            cb_model = CatBoostClassifier(
                iterations=4500,
                learning_rate=0.024,
                depth=6,
                l2_leaf_reg=12.0,
                nan_mode="Min",
                eval_metric="AUC",
                random_seed=400 + fold_idx,
                thread_count=-1,
                verbose=False
            )
            cb_model.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), early_stopping_rounds=100, verbose=False)
            p_va_cb = cb_model.predict_proba(X_va_f)[:, 1]
            p_te_cb = cb_model.predict_proba(X_te_f)[:, 1]
            oof_cb_sym[va_idx] = to_average_uniform_rank_f64(p_va_cb)
            tst_cb_sym += to_average_uniform_rank_f64(p_te_cb) / n_splits
            tst_cb_sym_prob += p_te_cb.astype(np.float64) / n_splits
        active_va = [oof_lgb_cons[va_idx], oof_lgb_deep[va_idx]]
        active_w = [0.35, 0.25]
        if HAS_XGBOOST:
            active_va.append(oof_xgb_d6[va_idx])
            active_w.append(0.25)
        if HAS_CATBOOST:
            active_va.append(oof_cb_sym[va_idx])
            active_w.append(0.15)
        w_f = np.array(active_w, dtype=np.float64)
        w_f /= np.sum(w_f)
        fold_blend = np.column_stack(active_va) @ w_f
        f_auc = float(roc_auc_score(y_va_f, fold_blend))
        print(f"Fold {fold_idx:02d}/{n_splits:02d} | Grand Synthesis 4-Family Validation ROC-AUC: {f_auc:.6f}", flush=True)
    oof_cols = [oof_lgb_cons, oof_lgb_deep]
    tst_cols = [tst_lgb_cons, tst_lgb_deep]
    prob_cols = [tst_lgb_cons_prob, tst_lgb_deep_prob]
    if HAS_XGBOOST:
        oof_cols.append(oof_xgb_d6)
        tst_cols.append(tst_xgb_d6)
        prob_cols.append(tst_xgb_d6_prob)
    if HAS_CATBOOST:
        oof_cols.append(oof_cb_sym)
        tst_cols.append(tst_cb_sym)
        prob_cols.append(tst_cb_sym_prob)
    oof_grand = np.column_stack(oof_cols)
    tst_grand = np.column_stack(tst_cols)
    prob_grand = np.column_stack(prob_cols)
    Z_oof = norm.ppf(np.clip(oof_grand, 1e-6, 1.0 - 1e-6))
    Z_tst = norm.ppf(np.clip(tst_grand, 1e-6, 1.0 - 1e-6))
    aucs_fam = np.array([roc_auc_score(y_tr, Z_oof[:, j]) for j in range(Z_oof.shape[1])], dtype=np.float64)
    d_fam = np.sqrt(2.0) * norm.ppf(np.clip(aucs_fam, 0.501, 0.999))
    cov_fam = 0.5 * np.cov(Z_oof, rowvar=False) + 0.5 * np.cov(Z_tst, rowvar=False) + np.eye(Z_oof.shape[1]) * 0.02
    w_internal = np.maximum(0.05, np.linalg.solve(cov_fam, d_fam))
    w_internal /= np.sum(w_internal)
    pooled_oof = Z_oof @ w_internal
    pooled_tst = Z_tst @ w_internal
    test_grand_prob = prob_grand @ w_internal
    oof_grand_rank = to_average_uniform_rank_f64(pooled_oof)
    test_grand_rank = to_strict_uniform_rank_f64(pooled_tst, tie_breaker_arr=test_grand_prob)
    print("=" * 84, flush=True)
    print(f"GRAND SYNTHESIS 4-FAMILY EARLY-FUSION OOF ROC-AUC: {roc_auc_score(y_tr, oof_grand_rank):.6f}", flush=True)
    print(f"Internal Family Weights: {np.round(w_internal, 4)}", flush=True)
    print("=" * 84, flush=True)
    np.savez_compressed(
        output_npz_path,
        oof_grand_595=oof_grand,
        test_grand_595=tst_grand,
        oof_grand_rank=oof_grand_rank,
        test_grand_rank=test_grand_rank,
        test_grand_prob=test_grand_prob,
        oof_lgb_cons=oof_lgb_cons,
        test_lgb_cons=tst_lgb_cons,
        test_lgb_cons_prob=tst_lgb_cons_prob,
        oof_lgb_deep=oof_lgb_deep,
        test_lgb_deep=tst_lgb_deep,
        test_lgb_deep_prob=tst_lgb_deep_prob,
        oof_xgb_d6=oof_xgb_d6,
        test_xgb_d6=tst_xgb_d6,
        test_xgb_d6_prob=tst_xgb_d6_prob,
        oof_cb_sym=oof_cb_sym,
        test_cb_sym=tst_cb_sym,
        test_cb_sym_prob=tst_cb_sym_prob,
        sk_id_curr_test=sk_id_test,
        feature_names=feat_names
    )

if __name__ == "__main__":
    main()
