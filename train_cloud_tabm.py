import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, norm

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_average_uniform_rank_f64(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).ravel()
    n = a.shape[0]
    return rankdata(a, method="average").astype(np.float64) / (n + 1.0)

def to_strict_uniform_rank_f64(primary_arr: np.ndarray, tie_breaker_arr: np.ndarray = None) -> np.ndarray:
    p = np.asarray(primary_arr, dtype=np.float64).ravel()
    n = p.shape[0]
    if tie_breaker_arr is not None:
        tb = np.asarray(tie_breaker_arr, dtype=np.float64).ravel()
        order = np.lexsort((tb, p))
    else:
        order = np.argsort(p, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    return ranks

def compute_infold_knn_manifold_features(
    X_tr_imp: np.ndarray,
    y_tr_fold: np.ndarray,
    X_va_imp: np.ndarray,
    X_te_imp: np.ndarray,
    top_col_indices: np.ndarray
):
    scaler = StandardScaler()
    Z_tr = np.clip(scaler.fit_transform(X_tr_imp[:, top_col_indices]), -6.0, 6.0).astype(np.float32)
    Z_va = np.clip(scaler.transform(X_va_imp[:, top_col_indices]), -6.0, 6.0).astype(np.float32)
    Z_te = np.clip(scaler.transform(X_te_imp[:, top_col_indices]), -6.0, 6.0).astype(np.float32)

    nn_model = NearestNeighbors(n_neighbors=501, algorithm="auto", metric="euclidean", n_jobs=-1)
    nn_model.fit(Z_tr)

    dist_tr, idx_tr = nn_model.kneighbors(Z_tr, n_neighbors=501)
    dist_tr = dist_tr[:, 1:]
    idx_tr = idx_tr[:, 1:]

    dist_va, idx_va = nn_model.kneighbors(Z_va, n_neighbors=500)
    dist_te, idx_te = nn_model.kneighbors(Z_te, n_neighbors=500)

    def summarize_neighbors(dists: np.ndarray, indices: np.ndarray) -> np.ndarray:
        y_neighbors = y_tr_fold[indices].astype(np.float32)
        weights_500 = 1.0 / (dists + 0.15)
        weights_500 /= weights_500.sum(axis=1, keepdims=True)
        w_mean_500 = (y_neighbors * weights_500).sum(axis=1, keepdims=True)
        u_mean_500 = y_neighbors.mean(axis=1, keepdims=True)
        u_mean_100 = y_neighbors[:, :100].mean(axis=1, keepdims=True)
        u_mean_30 = y_neighbors[:, :30].mean(axis=1, keepdims=True)
        mean_dist_50 = np.log1p(dists[:, :50].mean(axis=1, keepdims=True))
        return np.hstack([w_mean_500, u_mean_500, u_mean_100, u_mean_30, mean_dist_50]).astype(np.float32)

    knn_tr = summarize_neighbors(dist_tr, idx_tr)
    knn_va = summarize_neighbors(dist_va, idx_va)
    knn_te = summarize_neighbors(dist_te, idx_te)
    return knn_tr, knn_va, knn_te

class HybridPiecewisePeriodicEmbedding(nn.Module):
    def __init__(self, n_features: int, n_frequencies: int = 6, sigma: float = 0.05):
        super().__init__()
        self.n_features = n_features
        self.freqs = nn.Parameter(torch.randn(1, n_features, n_frequencies) * sigma)
        self.phases = nn.Parameter(torch.zeros(1, n_features, n_frequencies))
        self.proj_w = nn.Parameter(
            torch.randn(n_features, n_frequencies * 2, 4) * (1.0 / np.sqrt(n_frequencies * 2))
        )
        self.proj_b = nn.Parameter(torch.zeros(1, n_features, 4))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * np.pi * x.unsqueeze(-1) * self.freqs + self.phases
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        emb = self.act(torch.einsum("bfm,fmc->bfc", fourier, self.proj_w) + self.proj_b)
        out = torch.cat([x.unsqueeze(-1), emb], dim=-1)
        return out.reshape(x.size(0), -1)

class LinearBatchEnsemble(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 32, first_layer: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.r_adapter = nn.Parameter(torch.empty(k, in_features))
        self.s_adapter = nn.Parameter(torch.empty(k, out_features))
        self.bias = nn.Parameter(torch.zeros(k, out_features))
        self.reset_parameters(first_layer=first_layer)

    def reset_parameters(self, first_layer: bool) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if first_layer:
            signs = torch.randint(0, 2, (self.k, self.in_features), dtype=torch.float32) * 2.0 - 1.0
            jitter = torch.randn(self.k, self.in_features) * 0.08 + 1.0
            self.r_adapter.data.copy_(signs * jitter)
            nn.init.ones_(self.s_adapter)
        else:
            nn.init.ones_(self.r_adapter)
            nn.init.ones_(self.s_adapter)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x * self.r_adapter.unsqueeze(0)
        xw = torch.matmul(xr, self.weight)
        return xw * self.s_adapter.unsqueeze(0) + self.bias.unsqueeze(0)

class IndependentEnsembleReadout(nn.Module):
    def __init__(self, in_features: int, k: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k, in_features) * (1.0 / np.sqrt(in_features)))
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias.unsqueeze(0)

class TabMResidualBlock(nn.Module):
    def __init__(self, d_model: int, k: int = 32, dropout: float = 0.22):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.be1 = LinearBatchEnsemble(d_model, d_model, k=k, first_layer=False)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.be2 = LinearBatchEnsemble(d_model, d_model, k=k, first_layer=False)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(max(0.10, dropout - 0.05))
        self.res_scale = nn.Parameter(torch.full((1, k, d_model), 0.20))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(self.act1(self.be1(self.norm1(x))))
        h = self.drop2(self.act2(self.be2(self.norm2(h))))
        return x + self.res_scale * h

class TabM32Architecture(nn.Module):
    def __init__(self, in_raw_dim: int, in_aux_dim: int, k: int = 32, d_model: int = 320, n_blocks: int = 3):
        super().__init__()
        self.in_raw_dim = in_raw_dim
        self.k = k
        self.num_emb = HybridPiecewisePeriodicEmbedding(in_raw_dim, n_frequencies=6, sigma=0.05)
        total_in_dim = in_raw_dim * 5 + in_aux_dim
        self.input_norm = nn.LayerNorm(total_in_dim)
        self.input_drop = nn.Dropout(0.10)
        self.stem_be = LinearBatchEnsemble(total_in_dim, d_model, k=k, first_layer=True)
        self.stem_act = nn.SiLU()
        self.stem_drop = nn.Dropout(0.18)
        self.blocks = nn.ModuleList([
            TabMResidualBlock(d_model=d_model, k=k, dropout=0.20 + 0.03 * i)
            for i in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_model * 2)
        self.head = IndependentEnsembleReadout(d_model * 2, k=k)

    def forward(self, x: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        if self.training and noise_std > 0.0:
            gauss_x = x[:, :self.in_raw_dim] + torch.randn_like(x[:, :self.in_raw_dim]) * noise_std
            aux_x = x[:, self.in_raw_dim:]
        else:
            gauss_x = x[:, :self.in_raw_dim]
            aux_x = x[:, self.in_raw_dim:]
        emb_x = self.num_emb(gauss_x)
        full_x = self.input_drop(self.input_norm(torch.cat([emb_x, aux_x], dim=1)))
        xk = full_x.unsqueeze(1).expand(-1, self.k, -1)
        stem_h = self.stem_drop(self.stem_act(self.stem_be(xk)))
        h = stem_h
        for blk in self.blocks:
            h = blk(h)
        cat_h = self.head_norm(torch.cat([h, stem_h], dim=-1))
        return self.head(cat_h)

class TabMIndependentLoss(nn.Module):
    def __init__(self, pos_weight: float = 2.0, smoothing: float = 0.004):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits_k: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        y_exp = y_smooth.unsqueeze(1).expand_as(logits_k)
        pw = torch.tensor(self.pos_weight, dtype=logits_k.dtype, device=logits_k.device)
        return F.binary_cross_entropy_with_logits(logits_k, y_exp, pos_weight=pw, reduction="mean")

def preprocess_fold_with_knn(
    X_tr_fold: np.ndarray,
    y_tr_fold: np.ndarray,
    X_va_fold: np.ndarray,
    X_te_full: np.ndarray,
    nan_cols_mask: np.ndarray,
    top_knn_cols: np.ndarray,
    seed: int
):
    X_tr_f32 = np.where(np.isinf(X_tr_fold), np.nan, X_tr_fold).astype(np.float32)
    X_va_f32 = np.where(np.isinf(X_va_fold), np.nan, X_va_fold).astype(np.float32)
    X_te_f32 = np.where(np.isinf(X_te_full), np.nan, X_te_full).astype(np.float32)

    tr_nan_ind = np.isnan(X_tr_f32[:, nan_cols_mask]).astype(np.float32)
    va_nan_ind = np.isnan(X_va_f32[:, nan_cols_mask]).astype(np.float32)
    te_nan_ind = np.isnan(X_te_f32[:, nan_cols_mask]).astype(np.float32)

    medians = np.nanmedian(X_tr_f32, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)

    X_tr_imp = np.where(np.isnan(X_tr_f32), medians, X_tr_f32)
    X_va_imp = np.where(np.isnan(X_va_f32), medians, X_va_f32)
    X_te_imp = np.where(np.isnan(X_te_f32), medians, X_te_f32)

    knn_tr, knn_va, knn_te = compute_infold_knn_manifold_features(
        X_tr_imp=X_tr_imp,
        y_tr_fold=y_tr_fold,
        X_va_imp=X_va_imp,
        X_te_imp=X_te_imp,
        top_col_indices=top_knn_cols
    )

    X_tr_aug = np.hstack([X_tr_imp, knn_tr]).astype(np.float32)
    X_va_aug = np.hstack([X_va_imp, knn_va]).astype(np.float32)
    X_te_aug = np.hstack([X_te_imp, knn_te]).astype(np.float32)

    qt = QuantileTransformer(
        n_quantiles=2000,
        output_distribution="normal",
        subsample=100000,
        random_state=seed
    )
    X_tr_qt = np.clip(qt.fit_transform(X_tr_aug), -5.0, 5.0).astype(np.float32)
    X_va_qt = np.clip(qt.transform(X_va_aug), -5.0, 5.0).astype(np.float32)
    X_te_qt = np.clip(qt.transform(X_te_aug), -5.0, 5.0).astype(np.float32)

    rs = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    X_tr_rs = np.arcsinh(np.clip(rs.fit_transform(X_tr_aug), -8.0, 8.0)).astype(np.float32)
    X_va_rs = np.arcsinh(np.clip(rs.transform(X_va_aug), -8.0, 8.0)).astype(np.float32)
    X_te_rs = np.arcsinh(np.clip(rs.transform(X_te_aug), -8.0, 8.0)).astype(np.float32)

    X_tr_out = np.hstack([X_tr_qt, X_tr_rs, tr_nan_ind]).astype(np.float32)
    X_va_out = np.hstack([X_va_qt, X_va_rs, va_nan_ind]).astype(np.float32)
    X_te_out = np.hstack([X_te_qt, X_te_rs, te_nan_ind]).astype(np.float32)

    return X_tr_out, X_va_out, X_te_out, X_tr_aug.shape[1]

@torch.no_grad()
def predict_tabm(model: nn.Module, X_tensor: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    n_samples = X_tensor.size(0)
    for start in range(0, n_samples, batch_size):
        xb = X_tensor[start:start + batch_size].to(device, non_blocking=True)
        logits_k = model(xb, noise_std=0.0)
        probs_k = torch.sigmoid(logits_k)
        mean_p = probs_k.mean(dim=1).detach().cpu().numpy().astype(np.float64)
        preds.append(mean_p)
    return np.concatenate(preds, axis=0)

def train_tabm_single_seed(
    X_tr_proc: np.ndarray,
    y_tr_fold: np.ndarray,
    X_va_proc: np.ndarray,
    y_va_fold: np.ndarray,
    X_te_proc: np.ndarray,
    in_raw_dim: int,
    device: torch.device,
    seed: int,
    k_heads: int = 32,
    max_epochs: int = 36,
    batch_size: int = 512,
    patience: int = 8,
    top_k_ckpts: int = 3
):
    set_all_seeds(seed)
    in_aux_dim = X_tr_proc.shape[1] - in_raw_dim
    train_ds = TensorDataset(
        torch.from_numpy(X_tr_proc),
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
    X_va_tensor = torch.from_numpy(X_va_proc)
    X_te_tensor = torch.from_numpy(X_te_proc)

    model = TabM32Architecture(
        in_raw_dim=in_raw_dim,
        in_aux_dim=in_aux_dim,
        k=k_heads,
        d_model=320,
        n_blocks=3
    ).to(device)

    criterion = TabMIndependentLoss(pos_weight=2.0, smoothing=0.004)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.4e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1.5e-5)

    best_auc = -1.0
    no_improve = 0
    top_checkpoints = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_k = model(xb, noise_std=0.012)
            loss = criterion(logits_k, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        va_preds = predict_tabm(model, X_va_tensor, batch_size=2048, device=device)
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

    va_rank_blend = np.zeros(X_va_proc.shape[0], dtype=np.float64)
    va_prob_blend = np.zeros(X_va_proc.shape[0], dtype=np.float64)
    te_rank_blend = np.zeros(X_te_proc.shape[0], dtype=np.float64)
    te_prob_blend = np.zeros(X_te_proc.shape[0], dtype=np.float64)

    for w, (_, state_dict, ckpt_va_preds) in zip(weights, top_checkpoints):
        model.load_state_dict(state_dict)
        ckpt_te_preds = predict_tabm(model, X_te_tensor, batch_size=2048, device=device)
        va_prob_blend += w * ckpt_va_preds
        va_rank_blend += w * to_average_uniform_rank_f64(ckpt_va_preds)
        te_prob_blend += w * ckpt_te_preds
        te_rank_blend += w * to_average_uniform_rank_f64(ckpt_te_preds)

    return va_rank_blend, va_prob_blend, te_rank_blend, te_prob_blend

def select_top_manifold_columns(X_tr: np.ndarray, y_tr: np.ndarray, k_cols: int = 16) -> np.ndarray:
    X_clean = np.where(np.isinf(X_tr), np.nan, X_tr).astype(np.float64)
    scores = np.zeros(X_clean.shape[1], dtype=np.float64)
    y_f = y_tr.astype(np.float64)
    y_centered = y_f - y_f.mean()
    y_norm = np.sqrt(np.sum(y_centered ** 2)) + 1e-12
    for j in range(X_clean.shape[1]):
        col = X_clean[:, j]
        valid = ~np.isnan(col)
        if valid.mean() < 0.25:
            continue
        c_val = col[valid]
        c_centered = c_val - c_val.mean()
        denom = np.sqrt(np.sum(c_centered ** 2)) * y_norm + 1e-12
        corr = abs(np.sum(c_centered * y_centered[valid]) / denom)
        scores[j] = corr
    top_idx = np.argsort(scores)[::-1][:k_cols]
    return top_idx

def main():
    candidate_data_paths = [
        "data/matrix_226_features_buhlmann_tier2.npz",
        "matrix_226_features_buhlmann_tier2.npz",
        "output/matrix_226_features_buhlmann_tier2.npz",
        "C:/Users/User/Documents/Personal Stuffs/GCI World/output/matrix_226_features_buhlmann_tier2.npz"
    ]
    input_npz_path = None
    for p in candidate_data_paths:
        if os.path.exists(p):
            input_npz_path = p
            break
    if input_npz_path is None:
        raise FileNotFoundError("Could not find matrix_226_features_buhlmann_tier2.npz in any expected location")

    candidate_resnet_paths = [
        "output/oof_and_test_tabular_neural_net.npz",
        "oof_and_test_tabular_neural_net.npz",
        "c:/Users/User/Downloads/oof_and_test_tabular_neural_net.npz",
        "C:/Users/User/Documents/Personal Stuffs/GCI World/output/oof_and_test_tabular_neural_net.npz"
    ]
    resnet_npz_path = None
    for p in candidate_resnet_paths:
        if os.path.exists(p):
            resnet_npz_path = p
            break

    os.makedirs("output", exist_ok=True)
    output_npz_path = "output/oof_and_test_tabm_32heads.npz"
    submission_csv_path = "output/submission_tabm_32heads_rank.csv"

    n_splits = 10
    cv_split_seed = 42
    model_seeds = [303, 404, 505]
    k_heads = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))

    data = np.load(input_npz_path, allow_pickle=True)
    X_tr = np.asarray(data["X_tr"], dtype=np.float32)
    X_te = np.asarray(data["X_te"], dtype=np.float32)
    y_tr = np.asarray(data["y_tr"], dtype=np.int64).ravel()
    feature_names = data["feature_names"] if "feature_names" in data.files else np.arange(X_tr.shape[1])
    sk_id_curr_test = data["sk_id_curr_test"] if "sk_id_curr_test" in data.files else np.arange(X_te.shape[0])

    n_train = X_tr.shape[0]
    n_test = X_te.shape[0]

    nan_cols_mask = (np.isnan(X_tr).mean(axis=0) > 0.001) | (np.isnan(X_te).mean(axis=0) > 0.001)
    top_knn_cols = select_top_manifold_columns(X_tr, y_tr, k_cols=14)

    print(f"Device: {device} | TabM-{k_heads} ({len(model_seeds)} seeds) + In-Fold k-NN 500 Manifold Injection...")

    oof_tabm_rank = np.zeros(n_train, dtype=np.float64)
    oof_tabm_prob = np.zeros(n_train, dtype=np.float64)
    test_tabm_rank_accum = np.zeros(n_test, dtype=np.float64)
    test_tabm_prob_accum = np.zeros(n_test, dtype=np.float64)
    fold_aucs = np.zeros(n_splits, dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_split_seed)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr), start=1):
        X_tr_proc, X_va_proc, X_te_proc, aug_raw_dim = preprocess_fold_with_knn(
            X_tr[tr_idx],
            y_tr[tr_idx],
            X_tr[va_idx],
            X_te,
            nan_cols_mask=nan_cols_mask,
            top_knn_cols=top_knn_cols,
            seed=cv_split_seed + fold_idx
        )
        y_tr_fold = y_tr[tr_idx]
        y_va_fold = y_tr[va_idx]

        fold_va_rank = np.zeros(len(va_idx), dtype=np.float64)
        fold_va_prob = np.zeros(len(va_idx), dtype=np.float64)
        fold_te_rank = np.zeros(n_test, dtype=np.float64)
        fold_te_prob = np.zeros(n_test, dtype=np.float64)

        for m_seed in model_seeds:
            va_r, va_p, te_r, te_p = train_tabm_single_seed(
                X_tr_proc=X_tr_proc,
                y_tr_fold=y_tr_fold,
                X_va_proc=X_va_proc,
                y_va_fold=y_va_fold,
                X_te_proc=X_te_proc,
                in_raw_dim=aug_raw_dim,
                device=device,
                seed=m_seed + fold_idx * 1000,
                k_heads=k_heads,
                max_epochs=36,
                batch_size=512,
                patience=8,
                top_k_ckpts=3
            )
            fold_va_rank += to_average_uniform_rank_f64(va_r) / len(model_seeds)
            fold_va_prob += va_p / len(model_seeds)
            fold_te_rank += to_average_uniform_rank_f64(te_r) / len(model_seeds)
            fold_te_prob += te_p / len(model_seeds)

        fold_auc = float(roc_auc_score(y_va_fold, fold_va_rank))
        fold_aucs[fold_idx - 1] = fold_auc
        oof_tabm_rank[va_idx] = to_average_uniform_rank_f64(fold_va_rank)
        oof_tabm_prob[va_idx] = fold_va_prob

        test_tabm_rank_accum += to_average_uniform_rank_f64(fold_te_rank) / n_splits
        test_tabm_prob_accum += fold_te_prob / n_splits

        print(f"Fold {fold_idx:02d}/{n_splits:02d} | TabM-{k_heads} Validation ROC-AUC: {fold_auc:.6f}", flush=True)

    full_tabm_auc = float(roc_auc_score(y_tr, oof_tabm_rank))
    mean_fold_auc = float(np.mean(fold_aucs))
    test_tabm_strict = to_strict_uniform_rank_f64(test_tabm_rank_accum, tie_breaker_arr=test_tabm_prob_accum)

    print("-" * 74, flush=True)
    print(f"TabM-{k_heads} Mean 10-Fold ROC-AUC        : {mean_fold_auc:.6f}", flush=True)
    print(f"TabM-{k_heads} Full 10-Fold OOF Rank ROC-AUC: {full_tabm_auc:.6f}", flush=True)
    print(f"Unique Test Rank Values                 : {len(np.unique(test_tabm_strict))} / {n_test}", flush=True)

    oof_super_nn = oof_tabm_rank.copy()
    test_super_nn = test_tabm_strict.copy()

    if resnet_npz_path is not None and os.path.exists(resnet_npz_path):
        res_data = np.load(resnet_npz_path, allow_pickle=True)
        oof_res_u = to_average_uniform_rank_f64(res_data["oof_nn_rank"])
        test_res_u = to_strict_uniform_rank_f64(res_data["test_nn"], tie_breaker_arr=res_data["test_nn_prob"])
        res_auc = float(roc_auc_score(y_tr, oof_res_u))
        nn_corr = float(np.corrcoef(oof_tabm_rank, oof_res_u)[0, 1])

        best_alpha = 0.65
        best_super_auc = -1.0
        for alpha in np.linspace(0.0, 1.0, 201):
            cand = alpha * oof_tabm_rank + (1.0 - alpha) * oof_res_u
            c_auc = float(roc_auc_score(y_tr, cand))
            if c_auc > best_super_auc:
                best_super_auc = c_auc
                best_alpha = float(alpha)

        oof_super_nn = to_average_uniform_rank_f64(best_alpha * oof_tabm_rank + (1.0 - best_alpha) * oof_res_u)
        test_super_nn = to_strict_uniform_rank_f64(
            best_alpha * test_tabm_strict + (1.0 - best_alpha) * test_res_u,
            tie_breaker_arr=test_tabm_prob_accum
        )
        print(f"ResNet-PLR OOF AUC                      : {res_auc:.6f}", flush=True)
        print(f"Inter-Neural Correlation (TabM/ResNet)  : {nn_corr:.6f}", flush=True)
        lift_super = (best_super_auc - res_auc) * 10000.0
        print(f"Super-Neural OOF (alpha_TabM={best_alpha:.2f})          : {best_super_auc:.6f} ({lift_super:+.2f} bps)", flush=True)
        print("-" * 74, flush=True)

    np.savez_compressed(
        output_npz_path,
        oof_nn=oof_super_nn.astype(np.float64),
        oof_nn_rank=oof_super_nn.astype(np.float64),
        oof_nn_prob=oof_tabm_prob.astype(np.float64),
        oof_tabm_rank=oof_tabm_rank.astype(np.float64),
        test_nn=test_super_nn.astype(np.float64),
        test_nn_prob=test_tabm_prob_accum.astype(np.float64),
        test_tabm_rank=test_tabm_strict.astype(np.float64),
        fold_aucs=fold_aucs,
        sk_id_curr_test=sk_id_curr_test,
        feature_names=feature_names
    )

    sub_df = pd.DataFrame({
        "SK_ID_CURR": sk_id_curr_test,
        "TARGET": test_super_nn.astype(np.float64)
    })
    sub_df.to_csv(submission_csv_path, index=False, float_format="%.15f")
    print(f"Saved TabM + Super-Neural artifacts to {output_npz_path} and {submission_csv_path}", flush=True)

if __name__ == "__main__":
    main()
