import os
import sys
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

def to_average_uniform_rank_f64(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).ravel()
    valid = ~np.isnan(a)
    out = np.full_like(a, 0.5, dtype=np.float64)
    n_val = int(valid.sum())
    if n_val > 0:
        out[valid] = rankdata(a[valid], method="average").astype(np.float64) / (n_val + 1.0)
    return out

def to_strict_uniform_rank_f64(primary_arr: np.ndarray, tie_breaker_arr: np.ndarray) -> np.ndarray:
    p = np.asarray(primary_arr, dtype=np.float64).ravel()
    tb = np.asarray(tie_breaker_arr, dtype=np.float64).ravel()
    n = p.shape[0]
    order = np.lexsort((tb, p))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    return ranks

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_chunk_dirs = [
        "data",
        ".",
        "output",
        "../output",
        os.path.join(script_dir, "data"),
        script_dir,
        os.path.join(script_dir, "output")
    ]
    for d in candidate_chunk_dirs:
        pattern = os.path.join(d, "matrix_463_features_relational_tier4_part_*.bin")
        chunks = sorted(glob.glob(pattern))
        target_npz = os.path.join(d, "matrix_463_features_relational_tier4.npz")
        if chunks and not os.path.exists(target_npz):
            print(f"Reassembling {len(chunks)} chunks into {target_npz}...", flush=True)
            with open(target_npz, "wb") as f_out:
                for chunk in chunks:
                    with open(chunk, "rb") as f_in:
                        f_out.write(f_in.read())
            print(f"Reassembly complete: {target_npz} ({os.path.getsize(target_npz) / 1e6:.2f} MB)", flush=True)
    candidate_npz = [
        "data/matrix_463_features_relational_tier4.npz",
        "matrix_463_features_relational_tier4.npz",
        "output/matrix_463_features_relational_tier4.npz",
        "../output/matrix_463_features_relational_tier4.npz",
        os.path.join(script_dir, "data", "matrix_463_features_relational_tier4.npz"),
        os.path.join(script_dir, "matrix_463_features_relational_tier4.npz"),
        os.path.join(script_dir, "output", "matrix_463_features_relational_tier4.npz")
    ]
    for p in candidate_npz:
        if os.path.exists(p):
            print(f"Loading pre-compiled NPZ from {p}...", flush=True)
            d = np.load(p, allow_pickle=True)
            X_tr = np.asarray(d["X_tr"], dtype=np.float32)
            X_te = np.asarray(d["X_te"], dtype=np.float32)
            y_tr = np.asarray(d["y_tr"], dtype=np.int64).ravel()
            sk_id_test = np.asarray(d["sk_id_curr_test"]).ravel()
            cols = d["feature_names"] if "feature_names" in d.files else np.arange(X_tr.shape[1])
            return X_tr, X_te, y_tr, sk_id_test, cols
    candidate_tr_parquet = [
        "data/matrix_463_features_relational_tier4_train.parquet",
        "matrix_463_features_relational_tier4_train.parquet",
        "output/matrix_463_features_relational_tier4_train.parquet",
        "../output/matrix_463_features_relational_tier4_train.parquet"
    ]
    candidate_te_parquet = [
        "data/matrix_463_features_relational_tier4_test.parquet",
        "matrix_463_features_relational_tier4_test.parquet",
        "output/matrix_463_features_relational_tier4_test.parquet",
        "../output/matrix_463_features_relational_tier4_test.parquet"
    ]
    tr_p = next((p for p in candidate_tr_parquet if os.path.exists(p)), None)
    te_p = next((p for p in candidate_te_parquet if os.path.exists(p)), None)
    if tr_p and te_p:
        print(f"Loading Parquet from {tr_p} and {te_p}...", flush=True)
        df_tr = pd.read_parquet(tr_p)
        df_te = pd.read_parquet(te_p)
        drop_meta = [c for c in ["TARGET", "SK_ID_CURR", "fold", "index"] if c in df_tr.columns]
        feature_cols = [c for c in df_tr.columns if c not in drop_meta and c in df_te.columns]
        y_tr = df_tr["TARGET"].values.astype(np.int64).ravel() if "TARGET" in df_tr.columns else np.load("data/y.npy").ravel()
        sk_id_test = df_te["SK_ID_CURR"].values.ravel() if "SK_ID_CURR" in df_te.columns else np.arange(171202, 171202 + len(df_te))
        X_tr = np.empty((len(df_tr), len(feature_cols)), dtype=np.float32)
        X_te = np.empty((len(df_te), len(feature_cols)), dtype=np.float32)
        for j, col in enumerate(feature_cols):
            s_tr = df_tr[col]
            s_te = df_te[col]
            if str(s_tr.dtype) in ("object", "category", "string", "bool"):
                comb = pd.concat([s_tr.astype(str), s_te.astype(str)], axis=0)
                codes, _ = pd.factorize(comb)
                codes = codes.astype(np.float32)
                X_tr[:, j] = codes[:len(df_tr)]
                X_te[:, j] = codes[len(df_tr):]
            else:
                X_tr[:, j] = s_tr.to_numpy(dtype=np.float32, na_value=np.nan)
                X_te[:, j] = s_te.to_numpy(dtype=np.float32, na_value=np.nan)
        X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
        X_te = np.where(np.isinf(X_te), np.nan, X_te)
        return X_tr, X_te, y_tr, sk_id_test, np.array(feature_cols)
    raise FileNotFoundError("Could not find 463-feature matrix.")

def main():
    os.makedirs("output", exist_ok=True)
    candidate_dae = [
        "output/dae_128_latent_features.npz",
        "dae_128_latent_features.npz",
        "data/dae_128_latent_features.npz"
    ]
    dae_npz_path = next((p for p in candidate_dae if os.path.exists(p)), None)
    if not dae_npz_path:
        print("Error: dae_128_latent_features.npz not found. Run train_cloud_dae_tabm463.py first.", flush=True)
        sys.exit(1)
    output_npz_path = "output/oof_and_test_lgbm_dae491.npz"
    X_tr, X_te, y_tr, sk_id_test, _ = load_data()
    dae_data = np.load(dae_npz_path, allow_pickle=True)
    dae_tr = np.asarray(dae_data["dae_tr"], dtype=np.float32)
    dae_te = np.asarray(dae_data["dae_te"], dtype=np.float32)
    z_tr = dae_tr[:, :128]
    z_te = dae_te[:, :128]
    anom_tr = dae_tr[:, 128:]
    anom_te = dae_te[:, 128:]
    print("Fitting TruncatedSVD (24 components) on DAE latent space...", flush=True)
    svd = TruncatedSVD(n_components=24, random_state=42)
    svd_all = svd.fit_transform(np.vstack([z_tr, z_te])).astype(np.float32)
    svd_tr = svd_all[:len(X_tr)]
    svd_te = svd_all[len(X_tr):]
    X_tr_491 = np.hstack([X_tr, svd_tr, anom_tr]).astype(np.float32)
    X_te_491 = np.hstack([X_te, svd_te, anom_te]).astype(np.float32)
    print(f"Constructed 491-feature matrix: Train={X_tr_491.shape}, Test={X_te_491.shape}", flush=True)
    n_splits = 10
    seeds = [42, 2026]
    oof_rank = np.zeros(len(X_tr), dtype=np.float64)
    oof_prob = np.zeros(len(X_tr), dtype=np.float64)
    test_rank_accum = np.zeros(len(X_te), dtype=np.float64)
    test_prob_accum = np.zeros(len(X_te), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print("Starting 10-Fold LGBM-DAE-491 Training...", flush=True)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tr_491, y_tr), start=1):
        f_va_p = np.zeros(len(va_idx), dtype=np.float64)
        f_te_p = np.zeros(len(X_te), dtype=np.float64)
        for s in seeds:
            model = lgb.LGBMClassifier(
                n_estimators=3500,
                learning_rate=0.018,
                num_leaves=63,
                max_depth=6,
                min_child_samples=60,
                subsample=0.80,
                subsample_freq=1,
                colsample_bytree=0.35,
                reg_alpha=8.0,
                reg_lambda=15.0,
                random_state=s + fold_idx * 100,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(
                X_tr_491[tr_idx], y_tr[tr_idx],
                eval_set=[(X_tr_491[va_idx], y_tr[va_idx])],
                callbacks=[lgb.early_stopping(80, verbose=False)]
            )
            f_va_p += model.predict_proba(X_tr_491[va_idx])[:, 1] / len(seeds)
            f_te_p += model.predict_proba(X_te_491)[:, 1] / len(seeds)
        f_auc = float(roc_auc_score(y_tr[va_idx], f_va_p))
        oof_prob[va_idx] = f_va_p
        oof_rank[va_idx] = to_average_uniform_rank_f64(f_va_p)
        test_prob_accum += f_te_p / n_splits
        test_rank_accum += to_average_uniform_rank_f64(f_te_p) / n_splits
        print(f"Fold {fold_idx:02d}/{n_splits:02d} | LGBM-DAE-491 Validation ROC-AUC: {f_auc:.6f}", flush=True)
    full_auc = float(roc_auc_score(y_tr, oof_rank))
    test_strict = to_strict_uniform_rank_f64(test_rank_accum, tie_breaker_arr=test_prob_accum)
    print("=" * 78, flush=True)
    print(f"Full 10-Fold LGBM-DAE-491 OOF ROC-AUC: {full_auc:.6f}", flush=True)
    print("=" * 78, flush=True)
    np.savez_compressed(
        output_npz_path,
        oof_lgb_dae_rank=oof_rank,
        oof_lgb_dae_prob=oof_prob,
        test_lgb_dae_rank=test_strict,
        test_lgb_dae_prob=test_prob_accum,
        sk_id_curr_test=sk_id_test
    )
    print(f"Saved LGBM-DAE-491 stream to {output_npz_path}", flush=True)

if __name__ == "__main__":
    main()
