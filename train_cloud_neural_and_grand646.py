import os
import sys
import copy
import glob
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, norm
from scipy.special import logit

from train_cloud_dae_tabm463 import (
    set_all_seeds,
    load_data,
    extract_deduplicated_nan_signatures,
    train_transductive_swap_dae,
    LearnableNaNPeriodicEmbedding,
    TabM32RelationalDAE,
    TabMIndependentLoss,
    build_decoupled_optimizer,
    predict_tabm_463,
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

HAS_CUDA = torch.cuda.is_available()

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

def compute_cross_domain_features(
    X_t4_tr: np.ndarray,
    X_t4_te: np.ndarray,
    X_t2_tr: np.ndarray,
    X_t2_te: np.ndarray,
    t4_names: np.ndarray,
    t2_names: np.ndarray
):
    X4 = np.vstack([X_t4_tr, X_t4_te]).astype(np.float64)
    X2 = np.vstack([X_t2_tr, X_t2_te]).astype(np.float64)
    n4 = {str(c): i for i, c in enumerate(t4_names)}
    n2 = {str(c): i for i, c in enumerate(t2_names)}

    e1 = X4[:, n4["EXT_SOURCE_1"]]
    e2 = X4[:, n4["EXT_SOURCE_2"]]
    e3 = X4[:, n4["EXT_SOURCE_3"]]
    p1 = np.nan_to_num(X2[:, n2["EXT_SOURCE_1_PRED"]], nan=0.50)
    p2 = np.nan_to_num(X2[:, n2["EXT_SOURCE_2_PRED"]], nan=0.50)
    p3 = np.nan_to_num(X2[:, n2["EXT_SOURCE_3_PRED"]], nan=0.50)

    e1_imp = np.where(np.isnan(e1), p1, 0.85 * e1 + 0.15 * p1)
    e2_imp = np.where(np.isnan(e2), p2, 0.90 * e2 + 0.10 * p2)
    e3_imp = np.where(np.isnan(e3), p3, 0.88 * e3 + 0.12 * p3)

    ext6 = 0.25 * e1_imp + 0.37 * e2_imp + 0.38 * e3_imp
    ext6_prod = e1_imp * e2_imp * e3_imp
    ext6_min = np.minimum(np.minimum(e1_imp, e2_imp), e3_imp)
    ext6_max = np.maximum(np.maximum(e1_imp, e2_imp), e3_imp)
    ext6_std = np.std(np.vstack([e1_imp, e2_imp, e3_imp]), axis=0)

    ext1_res = e1 - p1
    ext2_res = e2 - p2
    ext3_res = e3 - p3
    res_stack = np.vstack([ext1_res, ext2_res, ext3_res])
    valid_cnt = np.sum(~np.isnan(res_stack), axis=0)
    res_sum = np.nansum(res_stack, axis=0)
    ext_res_mean = np.where(valid_cnt > 0, res_sum / np.maximum(1, valid_cnt), np.nan)

    org_dev = np.nan_to_num(X2[:, n2["surg_XG_ORG_DEV"]], nan=0.0)
    occ_dev = np.nan_to_num(X2[:, n2["surg_XG_OCC_DEV"]], nan=0.0)
    r2_rate = np.nan_to_num(X2[:, n2["surg_R2_RATE"]], nan=0.28)
    r2_res = np.nan_to_num(X2[:, n2["surg_R2_RATE_RES"]], nan=0.0)
    solv = np.nan_to_num(X2[:, n2["merton_solvency_buffer"]], nan=-0.25)

    ins_late = np.nan_to_num(X4[:, n4["INS2_LATE_RATIO"]], nan=0.0)
    ins_rec_late = np.nan_to_num(X4[:, n4["INS2_RECENT_LATE_RATIO"]], nan=0.0)
    ins_under = np.nan_to_num(X4[:, n4["INS2_UNDER_RATIO"]], nan=0.0)
    ins_rec_under = np.nan_to_num(X4[:, n4["INS2_RECENT_UNDER_RATIO"]], nan=0.0)
    cc_util = np.nan_to_num(X4[:, n4["CC2_UTIL_MEAN"]], nan=0.0)
    bur_ovd_r = np.nan_to_num(X4[:, n4["BUR2_OVERDUE_RATIO"]], nan=0.0)
    bur_act_d = np.nan_to_num(X4[:, n4["BUR2_ACTIVE_DEBT_INCOME"]], nan=0.0)
    app_ref_r = np.nan_to_num(X4[:, n4["APP2_R365_REFUSED_RATIO"]], nan=0.0)
    pos_dpd = np.nan_to_num(X4[:, n4["POS2_EVER_DPD_SHARE"]], nan=0.0)

    super_comp = (
        ext6
        + 0.12 * org_dev
        + 0.08 * occ_dev
        - 0.25 * r2_rate
        - 0.30 * ins_late
        - 0.15 * np.clip(cc_util, 0.0, 1.5)
        - 0.15 * np.clip(app_ref_r, 0.0, 1.0)
    )

    cd_cols = [
        ("CD_EXT6_CONSENSUS", ext6),
        ("CD_EXT6_PROD", ext6_prod),
        ("CD_EXT6_MIN", ext6_min),
        ("CD_EXT6_MAX", ext6_max),
        ("CD_EXT6_STD", ext6_std),
        ("CD_EXT1_IMP", e1_imp),
        ("CD_EXT2_IMP", e2_imp),
        ("CD_EXT3_IMP", e3_imp),
        ("CD_EXT1_RES", ext1_res),
        ("CD_EXT2_RES", ext2_res),
        ("CD_EXT3_RES", ext3_res),
        ("CD_EXT_RES_MEAN", ext_res_mean),
        ("CD_EXT6_PLUS_ORG_OCC", ext6 + 0.15 * org_dev + 0.10 * occ_dev),
        ("CD_EXT6_MINUS_R2", ext6 - 0.35 * r2_rate),
        ("CD_EXT6_MINUS_R2_RES", ext6 - 0.30 * r2_res),
        ("CD_EXT6_PLUS_SOLV", ext6 + 0.12 * solv),
        ("CD_EXT6_MINUS_INS_LATE", ext6 - 0.35 * ins_late - 0.25 * ins_rec_late),
        ("CD_EXT6_MINUS_CC_UTIL", ext6 - 0.18 * np.clip(cc_util, 0.0, 2.0)),
        ("CD_EXT6_MINUS_BUR_OVD", ext6 - 0.40 * np.clip(bur_ovd_r, 0.0, 1.0) - 0.03 * np.log1p(np.maximum(0.0, bur_act_d))),
        ("CD_EXT6_MINUS_APP_REF", ext6 - 0.18 * np.clip(app_ref_r, 0.0, 1.0)),
        ("CD_EXT6_MINUS_POS_DPD", ext6 - 0.20 * np.clip(pos_dpd, 0.0, 1.0)),
        ("CD_INS_LATE_ACCEL", ins_rec_late - ins_late),
        ("CD_INS_UNDER_ACCEL", ins_rec_under - ins_under),
        ("CD_SUPER_COMPOSITE", super_comp)
    ]
    cd_names = [name for name, _ in cd_cols]
    X_cd_all = np.column_stack([arr for _, arr in cd_cols]).astype(np.float32)
    n_tr = len(X_t4_tr)
    return X_cd_all[:n_tr], X_cd_all[n_tr:], cd_names, super_comp.astype(np.float32)

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
    soft_t0_pairs = P2[P2["tier"] == 0][["row", "row_p", "d", "ad"]].reset_index(drop=True)
    dyn_names = cat_names + [
        "FOLD_UIDY_N", "FOLD_UIDY_SUM", "FOLD_UIDY_NEAR_Y", "FOLD_UIDY_NEAR_D", "FOLD_UIDY_ANYBAD",
        "FOLD_UID2Y_T1_N", "FOLD_UID2Y_T1_SUM", "FOLD_UID2Y_T1_NEAR_Y", "FOLD_UID2Y_T1_NEAR_D", "FOLD_UID2Y_T1_ANYBAD",
        "FOLD_UID2Y_T0_N", "FOLD_UID2Y_T0_SUM", "FOLD_UID2Y_T0_NEAR_Y", "FOLD_UID2Y_T0_NEAR_D", "FOLD_UID2Y_T0_ANYBAD",
        "FOLD_BUH_TARIFF_LO", "FOLD_BUH_DEMO_LO", "FOLD_BUH_TOTAL_LO", "FOLD_MASTER_PRIOR"
    ]
    return code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, soft_t0_pairs, dyn_names, n_all

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
    F_df = pd.DataFrame({
        "N": g.size(),
        "SUM": g["yp"].sum(),
        "NEAR_Y": near["yp"],
        "NEAR_D": near["ad"]
    }).reindex(rows)
    F_df["ANYBAD"] = np.where(F_df["N"].isna(), np.nan, (F_df["SUM"] > 0).astype(np.float32))
    return F_df.to_numpy(dtype=np.float32)

def build_composite_priors(
    buh_mat: np.ndarray,
    uidy_mat: np.ndarray,
    uid1_mat: np.ndarray,
    super_comp_slice: np.ndarray
) -> np.ndarray:
    buh_clip = np.clip(buh_mat.astype(np.float64), 1e-4, 1.0 - 1e-4)
    lo = logit(buh_clip)
    demo_idx = [3, 4, 5, 6, 7, 8, 9, 10]
    tariff_idx = [0, 1, 2, 11, 12, 13, 14, 15, 16, 17]
    tariff_lo = np.mean(lo[:, tariff_idx], axis=1)
    demo_lo = np.mean(lo[:, demo_idx], axis=1)
    tot_lo = 0.60 * tariff_lo + 0.40 * demo_lo
    uid_near = np.nan_to_num(uidy_mat[:, 2].astype(np.float64), nan=0.0807)
    uid1_near = np.nan_to_num(uid1_mat[:, 2].astype(np.float64), nan=0.0807)
    master_prior = (
        -3.20 * super_comp_slice.astype(np.float64)
        + 0.85 * tot_lo
        + 1.10 * (uid_near - 0.0807)
        + 0.45 * (uid1_near - 0.0807)
    )
    return np.column_stack([tariff_lo, demo_lo, tot_lo, master_prior]).astype(np.float32)

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
    soft_t0_pairs: pd.DataFrame,
    super_comp_all: np.ndarray,
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
        enc_fold_tr = calc_buhlmann_vector(code_matrix[tr_idx, col_j], y_f_tr, n_sz)
        va_buh[:, col_j] = enc_fold_tr[code_matrix[va_idx, col_j]]
        enc_full_tr = calc_buhlmann_vector(code_matrix[:n_train, col_j], y_tr_full, n_sz)
        te_buh[:, col_j] = enc_full_tr[code_matrix[n_train:, col_j]]

    lab_fold = np.full(n_all, np.nan, dtype=np.float64)
    lab_fold[tr_idx] = y_f_tr.astype(np.float64)
    lab_full_train = np.full(n_all, np.nan, dtype=np.float64)
    lab_full_train[:n_train] = y_tr_full.astype(np.float64)
    te_idx = np.arange(n_train, n_all, dtype=np.int64)

    uidy_tr = compute_uid_target_block(tr_idx, lab_fold, confirmed_pairs)
    uidy_va = compute_uid_target_block(va_idx, lab_fold, confirmed_pairs)
    uidy_te = compute_uid_target_block(te_idx, lab_full_train, confirmed_pairs)

    uid1_tr = compute_uid_target_block(tr_idx, lab_fold, soft_t1_pairs)
    uid1_va = compute_uid_target_block(va_idx, lab_fold, soft_t1_pairs)
    uid1_te = compute_uid_target_block(te_idx, lab_full_train, soft_t1_pairs)

    uid0_tr = compute_uid_target_block(tr_idx, lab_fold, soft_t0_pairs)
    uid0_va = compute_uid_target_block(va_idx, lab_fold, soft_t0_pairs)
    uid0_te = compute_uid_target_block(te_idx, lab_full_train, soft_t0_pairs)

    comp_tr = build_composite_priors(tr_buh, uidy_tr, uid1_tr, super_comp_all[tr_idx])
    comp_va = build_composite_priors(va_buh, uidy_va, uid1_va, super_comp_all[va_idx])
    comp_te = build_composite_priors(te_buh, uidy_te, uid1_te, super_comp_all[te_idx])

    dyn_tr = np.hstack([tr_buh, uidy_tr, uid1_tr, uid0_tr, comp_tr]).astype(np.float32)
    dyn_va = np.hstack([va_buh, uidy_va, uid1_va, uid0_va, comp_va]).astype(np.float32)
    dyn_te = np.hstack([te_buh, uidy_te, uid1_te, uid0_te, comp_te]).astype(np.float32)
    return dyn_tr, dyn_va, dyn_te

class SwiGLUResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.22):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.fc_gate = nn.Linear(d_model, d_model * 2)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.drop2 = nn.Dropout(max(0.10, dropout - 0.05))
        self.res_scale = nn.Parameter(torch.full((1, d_model), 0.22))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, b = self.fc_gate(h).chunk(2, dim=-1)
        h = self.drop1(F.silu(a) * b)
        h = self.drop2(F.silu(self.fc_out(self.norm2(h))))
        return x + self.res_scale * h

class CosineGatedResNetPLR(nn.Module):
    def __init__(self, n_raw_features: int, n_aux_features: int, d_model: int = 512, n_blocks: int = 4):
        super().__init__()
        self.nan_plr = LearnableNaNPeriodicEmbedding(n_raw_features, n_frequencies=4, sigma=0.05)
        total_in = n_raw_features * 4 + n_aux_features
        self.in_norm = nn.LayerNorm(total_in)
        self.in_drop = nn.Dropout(0.12)
        self.stem = nn.Sequential(
            nn.Linear(total_in, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.20)
        )
        self.blocks = nn.ModuleList([
            SwiGLUResidualBlock(d_model=d_model, dropout=0.22 + 0.03 * i)
            for i in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Linear(d_model * 2, 1)

    def forward(self, x_gauss: torch.Tensor, x_isnan: torch.Tensor, x_aux: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        if self.training and noise_std > 0.0:
            x_gauss = x_gauss + torch.randn_like(x_gauss) * noise_std * (1.0 - x_isnan)
        emb = self.nan_plr(x_gauss, x_isnan)
        full_in = self.in_drop(self.in_norm(torch.cat([emb, x_aux], dim=1)))
        stem_h = self.stem(full_in)
        h = stem_h
        for blk in self.blocks:
            h = blk(h)
        cat_h = self.head_norm(torch.cat([h, stem_h], dim=-1))
        return self.head(cat_h).squeeze(-1)

@torch.no_grad()
def predict_resnet_plr(
    model: nn.Module,
    X_qt: np.ndarray,
    X_isnan: np.ndarray,
    X_aux: np.ndarray,
    batch_size: int,
    device: torch.device
) -> np.ndarray:
    model.eval()
    preds = []
    n_samples = X_qt.shape[0]
    t_qt = torch.from_numpy(X_qt)
    t_nan = torch.from_numpy(X_isnan)
    t_aux = torch.from_numpy(X_aux)
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    for start in range(0, n_samples, batch_size):
        b_qt = t_qt[start:start + batch_size].to(device, non_blocking=True)
        b_nan = t_nan[start:start + batch_size].to(device, non_blocking=True)
        b_aux = t_aux[start:start + batch_size].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(b_qt, b_nan, b_aux, noise_std=0.0)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        preds.append(probs)
    return np.concatenate(preds, axis=0)

def train_resnet_plr_seed(
    tr_tuple: tuple,
    y_tr_fold: np.ndarray,
    va_tuple: tuple,
    y_va_fold: np.ndarray,
    te_tuple: tuple,
    device: torch.device,
    seed: int,
    max_epochs: int = 28,
    batch_size: int = 512,
    patience: int = 7,
    top_k_ckpts: int = 3
):
    set_all_seeds(seed)
    X_tr_qt, isnan_tr, aux_tr = tr_tuple
    X_va_qt, isnan_va, aux_va = va_tuple
    X_te_qt, isnan_te, aux_te = te_tuple
    n_raw = X_tr_qt.shape[1]
    n_aux = aux_tr.shape[1]
    train_ds = TensorDataset(
        torch.from_numpy(X_tr_qt),
        torch.from_numpy(isnan_tr),
        torch.from_numpy(aux_tr),
        torch.from_numpy(y_tr_fold.astype(np.float32))
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    model = CosineGatedResNetPLR(
        n_raw_features=n_raw,
        n_aux_features=n_aux,
        d_model=512,
        n_blocks=4
    ).to(device)
    optimizer = build_decoupled_optimizer(model, lr=1.4e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1.4e-3,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.12,
        div_factor=15.0,
        final_div_factor=100.0
    )
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    pw = torch.tensor(2.0, dtype=torch.float32, device=device)
    best_auc = -1.0
    no_improve = 0
    top_checkpoints = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for b_qt, b_nan, b_aux, yb in train_loader:
            b_qt = b_qt.to(device, non_blocking=True)
            b_nan = b_nan.to(device, non_blocking=True)
            b_aux = b_aux.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            y_smooth = yb * 0.996 + 0.002
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(b_qt, b_nan, b_aux, noise_std=0.012)
                loss = F.binary_cross_entropy_with_logits(logits, y_smooth, pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        va_preds = predict_resnet_plr(model, X_va_qt, isnan_va, aux_va, batch_size=2048, device=device)
        val_auc = float(roc_auc_score(y_va_fold, va_preds))
        state_copy = copy.deepcopy(model.state_dict())
        top_checkpoints.append((val_auc, state_copy, va_preds))
        top_checkpoints.sort(key=lambda item: item[0], reverse=True)
        if len(top_checkpoints) > top_k_ckpts:
            top_checkpoints.pop()
        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    weights = np.array([0.50, 0.30, 0.20][:len(top_checkpoints)], dtype=np.float64)
    weights /= weights.sum()
    va_rank = np.zeros(X_va_qt.shape[0], dtype=np.float64)
    va_prob = np.zeros(X_va_qt.shape[0], dtype=np.float64)
    te_rank = np.zeros(X_te_qt.shape[0], dtype=np.float64)
    te_prob = np.zeros(X_te_qt.shape[0], dtype=np.float64)
    for w, (_, state_dict, ckpt_va_preds) in zip(weights, top_checkpoints):
        model.load_state_dict(state_dict)
        ckpt_te_preds = predict_resnet_plr(model, X_te_qt, isnan_te, aux_te, batch_size=2048, device=device)
        va_prob += w * ckpt_va_preds
        va_rank += w * to_average_uniform_rank_f64(ckpt_va_preds)
        te_prob += w * ckpt_te_preds
        te_rank += w * to_average_uniform_rank_f64(ckpt_te_preds)
    return va_rank, va_prob, te_rank, te_prob

def preprocess_fold_grand(
    X_tr_fold: np.ndarray,
    X_va_fold: np.ndarray,
    X_te_full: np.ndarray,
    dae_tr_fold: np.ndarray,
    dae_va_fold: np.ndarray,
    dae_te_full: np.ndarray,
    nan_sig_tr_fold: np.ndarray,
    nan_sig_va_fold: np.ndarray,
    nan_sig_te_full: np.ndarray,
    seed: int
):
    isnan_tr = np.isnan(X_tr_fold).astype(np.float32)
    isnan_va = np.isnan(X_va_fold).astype(np.float32)
    isnan_te = np.isnan(X_te_full).astype(np.float32)
    medians = np.nanmedian(X_tr_fold, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)
    X_tr_imp = np.where(np.isnan(X_tr_fold), medians, X_tr_fold)
    X_va_imp = np.where(np.isnan(X_va_fold), medians, X_va_fold)
    X_te_imp = np.where(np.isnan(X_te_full), medians, X_te_full)
    qt = QuantileTransformer(n_quantiles=2500, output_distribution="normal", subsample=110000, random_state=seed)
    X_tr_qt = np.clip(qt.fit_transform(X_tr_imp), -4.5, 4.5).astype(np.float32)
    X_va_qt = np.clip(qt.transform(X_va_imp), -4.5, 4.5).astype(np.float32)
    X_te_qt = np.clip(qt.transform(X_te_imp), -4.5, 4.5).astype(np.float32)
    rs = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    X_tr_rs = np.arcsinh(np.clip(rs.fit_transform(X_tr_imp), -8.0, 8.0)).astype(np.float32)
    X_va_rs = np.arcsinh(np.clip(rs.transform(X_va_imp), -8.0, 8.0)).astype(np.float32)
    X_te_rs = np.arcsinh(np.clip(rs.transform(X_te_imp), -8.0, 8.0)).astype(np.float32)
    aux_tr = np.hstack([X_tr_rs, dae_tr_fold, nan_sig_tr_fold]).astype(np.float32)
    aux_va = np.hstack([X_va_rs, dae_va_fold, nan_sig_va_fold]).astype(np.float32)
    aux_te = np.hstack([X_te_rs, dae_te_full, nan_sig_te_full]).astype(np.float32)
    return (X_tr_qt, isnan_tr, aux_tr), (X_va_qt, isnan_va, aux_va), (X_te_qt, isnan_te, aux_te)

def build_grand_synthesis_646_base(device: torch.device):
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

    X_cd_tr, X_cd_te, cd_names, super_comp_all = compute_cross_domain_features(
        X_t4_tr, X_t4_te, X_t2_tr, X_t2_te, t4_cols, t2_names
    )

    code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, soft_t0_pairs, dyn_names, n_all = precompute_tariff_and_entity_structures(
        X_t2_tr, X_t2_te, t2_names
    )

    nan_sig_tr, nan_sig_te, _ = extract_deduplicated_nan_signatures(X_t4_tr, X_t4_te)

    dae_cands = [
        "output/dae_128_latent_features.npz",
        "dae_128_latent_features.npz",
        "data/dae_128_latent_features.npz",
        os.path.join(script_dir, "output", "dae_128_latent_features.npz"),
        os.path.join(script_dir, "..", "output", "dae_128_latent_features.npz"),
        os.path.join(script_dir, "data", "dae_128_latent_features.npz")
    ]
    dae_path = next((p for p in dae_cands if os.path.exists(p)), None)
    if dae_path is not None:
        dd = np.load(dae_path, allow_pickle=True)
        dae_tr_132 = np.asarray(dd["dae_tr"], dtype=np.float32)
        dae_te_132 = np.asarray(dd["dae_te"], dtype=np.float32)
    else:
        os.makedirs("output", exist_ok=True)
        dae_tr_132, dae_te_132 = train_transductive_swap_dae(
            X_tr=X_t4_tr,
            X_te=X_t4_te,
            nan_sig_tr=nan_sig_tr,
            nan_sig_te=nan_sig_te,
            device=device,
            latent_dim=128,
            swap_prob=0.15,
            epochs=24,
            batch_size=1024,
            seed=42
        )
        np.savez_compressed("output/dae_128_latent_features.npz", dae_tr=dae_tr_132, dae_te=dae_te_132, sk_id_curr_test=sk_id_test)

    svd = TruncatedSVD(n_components=24, random_state=42)
    svd_all = svd.fit_transform(np.vstack([dae_tr_132[:, :128], dae_te_132[:, :128]])).astype(np.float32)
    svd_tr = svd_all[:len(X_t4_tr)]
    svd_te = svd_all[len(X_t4_tr):]
    anom_tr = dae_tr_132[:, 128:]
    anom_te = dae_te_132[:, 128:]
    X_dae_svd_tr = np.hstack([svd_tr, anom_tr]).astype(np.float32)
    X_dae_svd_te = np.hstack([svd_te, anom_te]).astype(np.float32)
    dae_names = [f"DAE_SVD_{i:02d}" for i in range(24)] + [f"DAE_ANOM_{i:02d}" for i in range(anom_tr.shape[1])]

    X_raw_static_tr = np.hstack([X_t4_tr, X_orth_tr, X_cd_tr]).astype(np.float32)
    X_raw_static_te = np.hstack([X_t4_te, X_orth_te, X_cd_te]).astype(np.float32)

    X_gbdt_base_tr = np.hstack([X_t4_tr, X_orth_tr, X_cd_tr, X_dae_svd_tr]).astype(np.float32)
    X_gbdt_base_te = np.hstack([X_t4_te, X_orth_te, X_cd_te, X_dae_svd_te]).astype(np.float32)

    all_feature_names = np.array(list(t4_cols) + orth_t2_names + cd_names + dae_names + dyn_names)
    total_dim = X_gbdt_base_tr.shape[1] + len(dyn_names)

    print("=" * 88, flush=True)
    print(f"BUILT GRAND SYNTHESIS 646 EARLY-FUSION MATRIX: Train=({len(X_t4_tr)}, {total_dim}), Test=({len(X_t4_te)}, {total_dim})", flush=True)
    print(f"  -> Tier-4 Relational Features               : {X_t4_tr.shape[1]}", flush=True)
    print(f"  -> Deduplicated Target-Free Tier-2 Features : {X_orth_tr.shape[1]}", flush=True)
    print(f"  -> Cross-Domain Interaction Features (CD)   : {X_cd_tr.shape[1]}", flush=True)
    print(f"  -> SVD-DAE & Anomaly Features               : {X_dae_svd_tr.shape[1]}", flush=True)
    print(f"  -> Fold-Isolated Dynamic Buhlmann & UIDTE   : {len(dyn_names)} (100% Full-Train Test Graph)", flush=True)
    print("=" * 88, flush=True)

    dyn_pack = (
        code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, soft_t0_pairs,
        super_comp_all, len(X_t4_tr), n_all
    )
    neural_aux_pack = (X_raw_static_tr, X_raw_static_te, dae_tr_132, dae_te_132, nan_sig_tr, nan_sig_te)
    return X_gbdt_base_tr, X_gbdt_base_te, y_tr, sk_id_test, all_feature_names, dyn_pack, neural_aux_pack

def run_phase1_neural_grand646(
    neural_aux_pack: tuple,
    dyn_pack: tuple,
    y_tr: np.ndarray,
    sk_id_test: np.ndarray,
    device: torch.device,
    neural_npz_path: str
):
    if os.path.exists(neural_npz_path):
        print(f"Phase 1 cached artifact found at {neural_npz_path}. Loading...", flush=True)
        d_n = np.load(neural_npz_path, allow_pickle=True)
        print(f"Loaded Neural Grand 646 OOF ROC-AUC: {roc_auc_score(y_tr, d_n['oof_neural646_rank']):.6f}", flush=True)
        return d_n

    X_raw_static_tr, X_raw_static_te, dae_tr_132, dae_te_132, nan_sig_tr, nan_sig_te = neural_aux_pack
    code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, soft_t0_pairs, super_comp_all, n_train, n_all = dyn_pack
    n_test = len(X_raw_static_te)
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_tabm_rank = np.zeros(n_train, dtype=np.float64)
    oof_tabm_prob = np.zeros(n_train, dtype=np.float64)
    tst_tabm_rank = np.zeros(n_test, dtype=np.float64)
    tst_tabm_prob = np.zeros(n_test, dtype=np.float64)

    oof_resnet_rank = np.zeros(n_train, dtype=np.float64)
    oof_resnet_prob = np.zeros(n_train, dtype=np.float64)
    tst_resnet_rank = np.zeros(n_test, dtype=np.float64)
    tst_resnet_prob = np.zeros(n_test, dtype=np.float64)

    print("=" * 88, flush=True)
    print("PHASE 1: TRAINING 10-FOLD DUAL-MANIFOLD NEURAL ZOO ON FULL GRAND 646 MATRIX", flush=True)
    print("=" * 88, flush=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_raw_static_tr, y_tr), start=1):
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
            soft_t0_pairs=soft_t0_pairs,
            super_comp_all=super_comp_all,
            fold_seed=1000 + fold_idx
        )
        X_tr_raw_f = np.hstack([X_raw_static_tr[tr_idx], dyn_tr]).astype(np.float32)
        X_va_raw_f = np.hstack([X_raw_static_tr[va_idx], dyn_va]).astype(np.float32)
        X_te_raw_f = np.hstack([X_raw_static_te, dyn_te]).astype(np.float32)

        tr_tup, va_tup, te_tup = preprocess_fold_grand(
            X_tr_raw_f, X_va_raw_f, X_te_raw_f,
            dae_tr_132[tr_idx], dae_tr_132[va_idx], dae_te_132,
            nan_sig_tr[tr_idx], nan_sig_tr[va_idx], nan_sig_te,
            seed=42 + fold_idx
        )
        y_tr_f = y_tr[tr_idx]
        y_va_f = y_tr[va_idx]

        vr_t1, vp_t1, tr_t1, tp_t1 = train_tabm_463_seed(
            tr_tuple=tr_tup,
            y_tr_fold=y_tr_f,
            va_tuple=va_tup,
            y_va_fold=y_va_f,
            te_tuple=te_tup,
            device=device,
            seed=707 + fold_idx * 1000,
            k_heads=32,
            max_epochs=30,
            batch_size=512,
            patience=7,
            top_k_ckpts=3
        )
        vr_t2, vp_t2, tr_t2, tp_t2 = train_tabm_463_seed(
            tr_tuple=tr_tup,
            y_tr_fold=y_tr_f,
            va_tuple=va_tup,
            y_va_fold=y_va_f,
            te_tuple=te_tup,
            device=device,
            seed=808 + fold_idx * 1000,
            k_heads=32,
            max_epochs=30,
            batch_size=512,
            patience=7,
            top_k_ckpts=3
        )
        f_tabm_va_r = 0.50 * to_average_uniform_rank_f64(vr_t1) + 0.50 * to_average_uniform_rank_f64(vr_t2)
        f_tabm_va_p = 0.50 * vp_t1 + 0.50 * vp_t2
        f_tabm_te_r = 0.50 * to_average_uniform_rank_f64(tr_t1) + 0.50 * to_average_uniform_rank_f64(tr_t2)
        f_tabm_te_p = 0.50 * tp_t1 + 0.50 * tp_t2

        oof_tabm_rank[va_idx] = to_average_uniform_rank_f64(f_tabm_va_r)
        oof_tabm_prob[va_idx] = f_tabm_va_p
        tst_tabm_rank += to_average_uniform_rank_f64(f_tabm_te_r) / n_splits
        tst_tabm_prob += f_tabm_te_p / n_splits

        vr_res, vp_res, tr_res, tp_res = train_resnet_plr_seed(
            tr_tuple=tr_tup,
            y_tr_fold=y_tr_f,
            va_tuple=va_tup,
            y_va_fold=y_va_f,
            te_tuple=te_tup,
            device=device,
            seed=909 + fold_idx * 1000,
            max_epochs=26,
            batch_size=512,
            patience=7,
            top_k_ckpts=3
        )
        oof_resnet_rank[va_idx] = to_average_uniform_rank_f64(vr_res)
        oof_resnet_prob[va_idx] = vp_res
        tst_resnet_rank += to_average_uniform_rank_f64(tr_res) / n_splits
        tst_resnet_prob += tp_res / n_splits

        f_blend = 0.72 * oof_tabm_rank[va_idx] + 0.28 * oof_resnet_rank[va_idx]
        print(
            f"Fold {fold_idx:02d}/{n_splits:02d} | TabM32-646: {roc_auc_score(y_va_f, oof_tabm_rank[va_idx]):.6f} | "
            f"SwiGLU-ResNet-646: {roc_auc_score(y_va_f, oof_resnet_rank[va_idx]):.6f} | "
            f"Neural646 Blend: {roc_auc_score(y_va_f, f_blend):.6f}",
            flush=True
        )

    z_tm_oof = norm.ppf(np.clip(oof_tabm_rank, 1e-6, 1.0 - 1e-6))
    z_rs_oof = norm.ppf(np.clip(oof_resnet_rank, 1e-6, 1.0 - 1e-6))
    z_tm_tst = norm.ppf(np.clip(to_average_uniform_rank_f64(tst_tabm_rank), 1e-6, 1.0 - 1e-6))
    z_rs_tst = norm.ppf(np.clip(to_average_uniform_rank_f64(tst_resnet_rank), 1e-6, 1.0 - 1e-6))

    oof_neural646_rank = to_average_uniform_rank_f64(0.72 * z_tm_oof + 0.28 * z_rs_oof)
    test_neural646_prob = 0.72 * tst_tabm_prob + 0.28 * tst_resnet_prob
    test_neural646_rank = to_strict_uniform_rank_f64(0.72 * z_tm_tst + 0.28 * z_rs_tst, tie_breaker_arr=test_neural646_prob)

    print("=" * 88, flush=True)
    print(f"PHASE 1 COMPLETE | TabM32-Grand646 Full OOF AUC       : {roc_auc_score(y_tr, oof_tabm_rank):.6f}", flush=True)
    print(f"PHASE 1 COMPLETE | SwiGLU-ResNet-Grand646 Full OOF AUC: {roc_auc_score(y_tr, oof_resnet_rank):.6f}", flush=True)
    print(f"PHASE 1 COMPLETE | Combined Neural-Grand646 OOF AUC   : {roc_auc_score(y_tr, oof_neural646_rank):.6f}", flush=True)
    print("=" * 88, flush=True)

    np.savez_compressed(
        neural_npz_path,
        oof_neural646_rank=oof_neural646_rank,
        test_neural646_rank=test_neural646_rank,
        test_neural646_prob=test_neural646_prob,
        oof_tabm646_rank=oof_tabm_rank,
        oof_tabm646_prob=oof_tabm_prob,
        test_tabm646_rank=to_strict_uniform_rank_f64(tst_tabm_rank, tie_breaker_arr=tst_tabm_prob),
        test_tabm646_prob=tst_tabm_prob,
        oof_resnet646_rank=oof_resnet_rank,
        oof_resnet646_prob=oof_resnet_prob,
        test_resnet646_rank=to_strict_uniform_rank_f64(tst_resnet_rank, tie_breaker_arr=tst_resnet_prob),
        test_resnet646_prob=tst_resnet_prob,
        sk_id_curr_test=sk_id_test
    )
    print(f"Saved Phase 1 Neural Grand 646 artifact to: {neural_npz_path}", flush=True)
    return np.load(neural_npz_path, allow_pickle=True)

def run_phase2_gbdt_grand646(
    X_gbdt_base_tr: np.ndarray,
    X_gbdt_base_te: np.ndarray,
    dyn_pack: tuple,
    y_tr: np.ndarray,
    sk_id_test: np.ndarray,
    feat_names: np.ndarray,
    gbdt_npz_path: str
):
    code_matrix, cat_sizes, confirmed_pairs, soft_t1_pairs, soft_t0_pairs, super_comp_all, n_train, n_all = dyn_pack
    n_test = len(X_gbdt_base_te)
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb_cons = np.zeros(n_train, dtype=np.float64)
    tst_lgb_cons = np.zeros(n_test, dtype=np.float64)
    tst_lgb_cons_prob = np.zeros(n_test, dtype=np.float64)

    oof_lgb_deep = np.zeros(n_train, dtype=np.float64)
    tst_lgb_deep = np.zeros(n_test, dtype=np.float64)
    tst_lgb_deep_prob = np.zeros(n_test, dtype=np.float64)

    oof_xgb_d6 = np.zeros(n_train, dtype=np.float64)
    tst_xgb_d6 = np.zeros(n_test, dtype=np.float64)
    tst_xgb_d6_prob = np.zeros(n_test, dtype=np.float64)

    oof_xgb_d4 = np.zeros(n_train, dtype=np.float64)
    tst_xgb_d4 = np.zeros(n_test, dtype=np.float64)
    tst_xgb_d4_prob = np.zeros(n_test, dtype=np.float64)

    oof_cb_sym = np.zeros(n_train, dtype=np.float64)
    tst_cb_sym = np.zeros(n_test, dtype=np.float64)
    tst_cb_sym_prob = np.zeros(n_test, dtype=np.float64)

    print("=" * 88, flush=True)
    print("PHASE 2: TRAINING 10-FOLD 5-FAMILY GBDT ZOO ON FULL GRAND 646 MATRIX", flush=True)
    print("=" * 88, flush=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_gbdt_base_tr, y_tr), start=1):
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
            soft_t0_pairs=soft_t0_pairs,
            super_comp_all=super_comp_all,
            fold_seed=1000 + fold_idx
        )
        X_tr_f = np.hstack([X_gbdt_base_tr[tr_idx], dyn_tr]).astype(np.float32)
        X_va_f = np.hstack([X_gbdt_base_tr[va_idx], dyn_va]).astype(np.float32)
        X_te_f = np.hstack([X_gbdt_base_te, dyn_te]).astype(np.float32)
        y_tr_f = y_tr[tr_idx]
        y_va_f = y_tr[va_idx]

        model_cons = lgb.LGBMClassifier(
            n_estimators=4500,
            learning_rate=0.015,
            num_leaves=48,
            max_depth=6,
            min_child_samples=70,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.30,
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
            n_estimators=4500,
            learning_rate=0.014,
            num_leaves=96,
            max_depth=7,
            min_child_samples=90,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.24,
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
            model_xgb6 = xgb.XGBClassifier(
                n_estimators=4500,
                learning_rate=0.016,
                max_depth=6,
                min_child_weight=55,
                subsample=0.80,
                colsample_bytree=0.26,
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
            model_xgb6.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
            p_va_xgb6 = model_xgb6.predict_proba(X_va_f)[:, 1]
            p_te_xgb6 = model_xgb6.predict_proba(X_te_f)[:, 1]
            oof_xgb_d6[va_idx] = to_average_uniform_rank_f64(p_va_xgb6)
            tst_xgb_d6 += to_average_uniform_rank_f64(p_te_xgb6) / n_splits
            tst_xgb_d6_prob += p_te_xgb6.astype(np.float64) / n_splits

            model_xgb4 = xgb.XGBClassifier(
                n_estimators=5000,
                learning_rate=0.020,
                max_depth=4,
                min_child_weight=80,
                subsample=0.82,
                colsample_bytree=0.34,
                reg_alpha=12.0,
                reg_lambda=30.0,
                tree_method="hist",
                device=xgb_device,
                eval_metric="auc",
                early_stopping_rounds=100,
                random_state=350 + fold_idx,
                n_jobs=-1,
                verbosity=0
            )
            model_xgb4.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
            p_va_xgb4 = model_xgb4.predict_proba(X_va_f)[:, 1]
            p_te_xgb4 = model_xgb4.predict_proba(X_te_f)[:, 1]
            oof_xgb_d4[va_idx] = to_average_uniform_rank_f64(p_va_xgb4)
            tst_xgb_d4 += to_average_uniform_rank_f64(p_te_xgb4) / n_splits
            tst_xgb_d4_prob += p_te_xgb4.astype(np.float64) / n_splits

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
        active_w = [0.30, 0.20]
        if HAS_XGBOOST:
            active_va.extend([oof_xgb_d6[va_idx], oof_xgb_d4[va_idx]])
            active_w.extend([0.22, 0.13])
        if HAS_CATBOOST:
            active_va.append(oof_cb_sym[va_idx])
            active_w.append(0.15)
        w_f = np.array(active_w, dtype=np.float64)
        w_f /= np.sum(w_f)
        fold_blend = np.column_stack(active_va) @ w_f
        f_auc = float(roc_auc_score(y_va_f, fold_blend))
        print(f"Fold {fold_idx:02d}/{n_splits:02d} | Grand Synthesis 646 5-Family Validation ROC-AUC: {f_auc:.6f}", flush=True)

    oof_cols = [oof_lgb_cons, oof_lgb_deep]
    tst_cols = [tst_lgb_cons, tst_lgb_deep]
    prob_cols = [tst_lgb_cons_prob, tst_lgb_deep_prob]
    if HAS_XGBOOST:
        oof_cols.extend([oof_xgb_d6, oof_xgb_d4])
        tst_cols.extend([tst_xgb_d6, tst_xgb_d4])
        prob_cols.extend([tst_xgb_d6_prob, tst_xgb_d4_prob])
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

    print("=" * 88, flush=True)
    print(f"PHASE 2 COMPLETE | GRAND 646 5-FAMILY GBDT OOF ROC-AUC: {roc_auc_score(y_tr, oof_grand_rank):.6f}", flush=True)
    print(f"Internal Family Weights: {np.round(w_internal, 4)}", flush=True)
    print("=" * 88, flush=True)

    np.savez_compressed(
        gbdt_npz_path,
        oof_grand_646=oof_grand,
        test_grand_646=tst_grand,
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
        oof_xgb_d4=oof_xgb_d4,
        test_xgb_d4=tst_xgb_d4,
        test_xgb_d4_prob=tst_xgb_d4_prob,
        oof_cb_sym=oof_cb_sym,
        test_cb_sym=tst_cb_sym,
        test_cb_sym_prob=tst_cb_sym_prob,
        sk_id_curr_test=sk_id_test,
        feature_names=feat_names
    )
    print(f"Saved Phase 2 GBDT Grand 646 artifact to: {gbdt_npz_path}", flush=True)
    return np.load(gbdt_npz_path, allow_pickle=True)

def main():
    os.makedirs("output", exist_ok=True)
    neural_npz_path = "output/oof_and_test_neural_grand_646.npz"
    gbdt_npz_path = "output/oof_and_test_grand_synthesis_646.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB", flush=True)

    X_gbdt_base_tr, X_gbdt_base_te, y_tr, sk_id_test, feat_names, dyn_pack, neural_aux_pack = build_grand_synthesis_646_base(device)

    d_neural = run_phase1_neural_grand646(
        neural_aux_pack=neural_aux_pack,
        dyn_pack=dyn_pack,
        y_tr=y_tr,
        sk_id_test=sk_id_test,
        device=device,
        neural_npz_path=neural_npz_path
    )

    d_gbdt = run_phase2_gbdt_grand646(
        X_gbdt_base_tr=X_gbdt_base_tr,
        X_gbdt_base_te=X_gbdt_base_te,
        dyn_pack=dyn_pack,
        y_tr=y_tr,
        sk_id_test=sk_id_test,
        feat_names=feat_names,
        gbdt_npz_path=gbdt_npz_path
    )

    z_n_oof = norm.ppf(np.clip(d_neural["oof_neural646_rank"], 1e-6, 1.0 - 1e-6))
    z_g_oof = norm.ppf(np.clip(d_gbdt["oof_grand_rank"], 1e-6, 1.0 - 1e-6))
    joint_oof = 0.68 * z_g_oof + 0.32 * z_n_oof
    print("=" * 88, flush=True)
    print(f"GRAND 646 JOINT (GBDT + NEURAL) EARLY-FUSION OOF ROC-AUC: {roc_auc_score(y_tr, joint_oof):.6f}", flush=True)
    print("=" * 88, flush=True)

if __name__ == "__main__":
    main()
