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
from sklearn.preprocessing import RobustScaler, QuantileTransformer
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

def to_uniform_rank_f64(arr: np.ndarray) -> np.ndarray:
    arr_f64 = np.asarray(arr, dtype=np.float64).ravel()
    n = arr_f64.shape[0]
    ranks = rankdata(arr_f64, method="average").astype(np.float64)
    return ranks / (n + 1.0)

def to_probit_rank_f64(arr: np.ndarray) -> np.ndarray:
    u = to_uniform_rank_f64(arr)
    u_clipped = np.clip(u, 1e-7, 1.0 - 1e-7)
    return norm.ppf(u_clipped).astype(np.float64)

class MishActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

class PeriodicNumericalEmbedding(nn.Module):
    def __init__(self, in_features: int, n_frequencies: int = 16, sigma: float = 0.08):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(in_features, n_frequencies) * sigma)
        self.projection = nn.Linear(in_features * n_frequencies * 2, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = 2.0 * np.pi * x.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        fourier = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        flat = fourier.reshape(x.size(0), -1)
        return self.act(self.norm(self.projection(flat)))

class TabularResNetBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, drop_first: float = 0.25, drop_second: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(drop_first)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.act2 = MishActivation()
        self.drop2 = nn.Dropout(drop_second)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.se_gate = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 8)),
            nn.SiLU(),
            nn.Linear(max(16, d_model // 8), d_model),
            nn.Sigmoid()
        )
        self.res_scale = nn.Parameter(torch.full((1, d_model), 0.15))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.drop1(self.act1(self.norm1(x))))
        h = self.fc2(self.drop2(self.act2(self.norm2(h))))
        h = h * self.se_gate(h)
        return x + self.res_scale * h

class DeepTabularResNet(nn.Module):
    def __init__(self, in_raw_dim: int, in_total_dim: int, d_model: int = 384, d_hidden: int = 512, n_blocks: int = 4):
        super().__init__()
        self.in_raw_dim = in_raw_dim
        self.plr = PeriodicNumericalEmbedding(in_raw_dim, n_frequencies=12, sigma=0.06)
        self.input_norm = nn.LayerNorm(in_total_dim + in_raw_dim)
        self.stem = nn.Sequential(
            nn.Dropout(0.10),
            nn.Linear(in_total_dim + in_raw_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            TabularResNetBlock(
                d_model=d_model,
                d_hidden=d_hidden,
                drop_first=0.25 + 0.02 * i,
                drop_second=0.15 + 0.02 * i
            )
            for i in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_model * 2)
        self.head_drop = nn.Dropout(0.20)
        self.head_fc = nn.Linear(d_model * 2, 1)

    def forward(self, x: torch.Tensor, training_noise: float = 0.0) -> torch.Tensor:
        if self.training and training_noise > 0.0:
            noise = torch.randn_like(x[:, :self.in_raw_dim * 2]) * training_noise
            x = torch.cat([x[:, :self.in_raw_dim * 2] + noise, x[:, self.in_raw_dim * 2:]], dim=1)
        x_gauss = x[:, :self.in_raw_dim]
        x_plr = self.plr(x_gauss)
        x_combined = self.input_norm(torch.cat([x, x_plr], dim=1))
        stem_out = self.stem(x_combined)
        h = stem_out
        for block in self.blocks:
            h = block(h)
        concat_repr = torch.cat([h, stem_out], dim=1)
        logits = self.head_fc(self.head_drop(self.head_norm(concat_repr)))
        return logits.squeeze(-1)

class SmoothWeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 2.25, smoothing: float = 0.005):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        pw = torch.tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, y_smooth, pos_weight=pw)

def preprocess_fold_matrices(
    X_tr_fold: np.ndarray,
    X_va_fold: np.ndarray,
    X_te_full: np.ndarray,
    nan_cols_mask: np.ndarray,
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

    qt = QuantileTransformer(
        n_quantiles=2000,
        output_distribution="normal",
        subsample=100000,
        random_state=seed
    )
    X_tr_qt = np.clip(qt.fit_transform(X_tr_imp), -5.0, 5.0).astype(np.float32)
    X_va_qt = np.clip(qt.transform(X_va_imp), -5.0, 5.0).astype(np.float32)
    X_te_qt = np.clip(qt.transform(X_te_imp), -5.0, 5.0).astype(np.float32)

    rs = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    X_tr_rs = np.clip(rs.fit_transform(X_tr_imp), -8.0, 8.0).astype(np.float32)
    X_va_rs = np.clip(rs.transform(X_va_imp), -8.0, 8.0).astype(np.float32)
    X_te_rs = np.clip(rs.transform(X_te_imp), -8.0, 8.0).astype(np.float32)

    X_tr_rs = np.arcsinh(X_tr_rs).astype(np.float32)
    X_va_rs = np.arcsinh(X_va_rs).astype(np.float32)
    X_te_rs = np.arcsinh(X_te_rs).astype(np.float32)

    X_tr_out = np.hstack([X_tr_qt, X_tr_rs, tr_nan_ind]).astype(np.float32)
    X_va_out = np.hstack([X_va_qt, X_va_rs, va_nan_ind]).astype(np.float32)
    X_te_out = np.hstack([X_te_qt, X_te_rs, te_nan_ind]).astype(np.float32)

    return X_tr_out, X_va_out, X_te_out

@torch.no_grad()
def predict_Batches(model: nn.Module, X_tensor: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    n_samples = X_tensor.size(0)
    for start in range(0, n_samples, batch_size):
        xb = X_tensor[start:start + batch_size].to(device, non_blocking=True)
        logits = model(xb, training_noise=0.0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        preds.append(probs)
    return np.concatenate(preds, axis=0)

def train_single_seed_fold(
    X_tr_proc: np.ndarray,
    y_tr_fold: np.ndarray,
    X_va_proc: np.ndarray,
    y_va_fold: np.ndarray,
    X_te_proc: np.ndarray,
    in_raw_dim: int,
    device: torch.device,
    seed: int,
    max_epochs: int = 38,
    batch_size: int = 512,
    patience: int = 9,
    top_k_checkpoints: int = 3
):
    set_all_seeds(seed)
    in_total_dim = X_tr_proc.shape[1]
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
    model = DeepTabularResNet(
        in_raw_dim=in_raw_dim,
        in_total_dim=in_total_dim,
        d_model=384,
        d_hidden=512,
        n_blocks=4
    ).to(device)
    criterion = SmoothWeightedBCELoss(pos_weight=2.25, smoothing=0.005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1.5e-5)
    best_auc = -1.0
    epochs_no_improve = 0
    top_checkpoints = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, training_noise=0.015)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        va_preds = predict_Batches(model, X_va_tensor, batch_size=2048, device=device)
        val_auc = float(roc_auc_score(y_va_fold, va_preds))
        state_copy = copy.deepcopy(model.state_dict())
        top_checkpoints.append((val_auc, state_copy, va_preds))
        top_checkpoints.sort(key=lambda item: item[0], reverse=True)
        if len(top_checkpoints) > top_k_checkpoints:
            top_checkpoints.pop()

        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    va_prob_blend = np.zeros(X_va_proc.shape[0], dtype=np.float64)
    va_rank_blend = np.zeros(X_va_proc.shape[0], dtype=np.float64)
    te_prob_blend = np.zeros(X_te_proc.shape[0], dtype=np.float64)
    te_rank_blend = np.zeros(X_te_proc.shape[0], dtype=np.float64)
    weights = np.array([0.50, 0.30, 0.20][:len(top_checkpoints)], dtype=np.float64)
    weights /= weights.sum()

    for w, (_, state_dict, ckpt_va_preds) in zip(weights, top_checkpoints):
        model.load_state_dict(state_dict)
        ckpt_te_preds = predict_Batches(model, X_te_tensor, batch_size=2048, device=device)
        va_prob_blend += w * ckpt_va_preds
        va_rank_blend += w * to_uniform_rank_f64(ckpt_va_preds)
        te_prob_blend += w * ckpt_te_preds
        te_rank_blend += w * to_uniform_rank_f64(ckpt_te_preds)

    auc_prob = float(roc_auc_score(y_va_fold, va_prob_blend))
    auc_rank = float(roc_auc_score(y_va_fold, va_rank_blend))
    if auc_prob >= auc_rank:
        return va_prob_blend, te_prob_blend, to_uniform_rank_f64(te_prob_blend)
    return va_rank_blend, te_prob_blend, to_uniform_rank_f64(te_rank_blend)

def main():
    candidate_paths = [
        "data/matrix_226_features_buhlmann_tier2.npz",
        "matrix_226_features_buhlmann_tier2.npz",
        "output/matrix_226_features_buhlmann_tier2.npz",
        "C:/Users/User/Documents/Personal Stuffs/GCI World/output/matrix_226_features_buhlmann_tier2.npz"
    ]
    input_npz_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            input_npz_path = p
            break
    if input_npz_path is None:
        raise FileNotFoundError("Could not find matrix_226_features_buhlmann_tier2.npz in any expected location")

    os.makedirs("output", exist_ok=True)
    output_npz_path = "output/oof_and_test_tabular_neural_net.npz"
    submission_csv_path = "output/submission_tabular_neural_net_rank.csv"

    n_splits = 10
    cv_split_seed = 42
    model_seeds = [101, 202]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))

    print(f"Device: {device} | Loading {input_npz_path}...")
    data = np.load(input_npz_path, allow_pickle=True)
    X_tr = np.asarray(data["X_tr"], dtype=np.float32)
    X_te = np.asarray(data["X_te"], dtype=np.float32)
    y_tr = np.asarray(data["y_tr"], dtype=np.int64).ravel()
    feature_names = data["feature_names"] if "feature_names" in data.files else np.arange(X_tr.shape[1])
    sk_id_curr_test = data["sk_id_curr_test"] if "sk_id_curr_test" in data.files else np.arange(X_te.shape[0])

    n_train, in_raw_dim = X_tr.shape
    n_test = X_te.shape[0]

    nan_rate_tr = np.isnan(X_tr).mean(axis=0)
    nan_rate_te = np.isnan(X_te).mean(axis=0)
    nan_cols_mask = (nan_rate_tr > 0.001) | (nan_rate_te > 0.001)

    print(f"Train shape: {X_tr.shape} | Test shape: {X_te.shape} | Positive rate: {y_tr.mean():.5f} | Informative NaN cols: {nan_cols_mask.sum()}")

    oof_nn_prob = np.zeros(n_train, dtype=np.float64)
    oof_nn_rank = np.zeros(n_train, dtype=np.float64)
    test_nn_prob = np.zeros(n_test, dtype=np.float64)
    test_nn_rank_accum = np.zeros(n_test, dtype=np.float64)
    test_nn_probit_accum = np.zeros(n_test, dtype=np.float64)
    fold_aucs = np.zeros(n_splits, dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_split_seed)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr), start=1):
        print(f"\n--- Starting Fold {fold_idx}/{n_splits} ---", flush=True)
        X_tr_proc, X_va_proc, X_te_proc = preprocess_fold_matrices(
            X_tr[tr_idx],
            X_tr[va_idx],
            X_te,
            nan_cols_mask=nan_cols_mask,
            seed=cv_split_seed + fold_idx
        )
        y_tr_fold = y_tr[tr_idx]
        y_va_fold = y_tr[va_idx]

        fold_va_pred = np.zeros(len(va_idx), dtype=np.float64)
        fold_te_prob = np.zeros(n_test, dtype=np.float64)
        fold_te_rank = np.zeros(n_test, dtype=np.float64)

        for s_idx, m_seed in enumerate(model_seeds):
            va_p, te_p, te_r = train_single_seed_fold(
                X_tr_proc=X_tr_proc,
                y_tr_fold=y_tr_fold,
                X_va_proc=X_va_proc,
                y_va_fold=y_va_fold,
                X_te_proc=X_te_proc,
                in_raw_dim=in_raw_dim,
                device=device,
                seed=m_seed + fold_idx * 1000,
                max_epochs=38,
                batch_size=512,
                patience=9,
                top_k_checkpoints=3
            )
            fold_va_pred += va_p / len(model_seeds)
            fold_te_prob += te_p / len(model_seeds)
            fold_te_rank += te_r / len(model_seeds)

        fold_auc = float(roc_auc_score(y_va_fold, fold_va_pred))
        fold_aucs[fold_idx - 1] = fold_auc
        oof_nn_prob[va_idx] = fold_va_pred
        oof_nn_rank[va_idx] = to_uniform_rank_f64(fold_va_pred)

        test_nn_prob += fold_te_prob / n_splits
        test_nn_rank_accum += to_uniform_rank_f64(fold_te_rank) / n_splits
        test_nn_probit_accum += to_probit_rank_f64(fold_te_rank) / n_splits

        print(f"Fold {fold_idx:02d}/{n_splits:02d} | Validation ROC-AUC: {fold_auc:.6f}", flush=True)

    overall_oof_prob_auc = float(roc_auc_score(y_tr, oof_nn_prob))
    overall_oof_rank_auc = float(roc_auc_score(y_tr, oof_nn_rank))
    mean_fold_auc = float(np.mean(fold_aucs))

    if overall_oof_prob_auc >= overall_oof_rank_auc:
        oof_nn = oof_nn_prob.astype(np.float64)
    else:
        oof_nn = oof_nn_rank.astype(np.float64)

    test_nn_raw_rank = to_uniform_rank_f64(test_nn_rank_accum)
    test_nn_probit_rank = to_uniform_rank_f64(test_nn_probit_accum)
    test_nn = to_uniform_rank_f64(0.5 * test_nn_raw_rank + 0.5 * test_nn_probit_rank).astype(np.float64)
    oof_nn_rank_norm = to_uniform_rank_f64(oof_nn).astype(np.float64)

    print("-" * 72, flush=True)
    print(f"Mean 10-Fold ROC-AUC        : {mean_fold_auc:.6f}", flush=True)
    print(f"Full OOF Probability ROC-AUC : {overall_oof_prob_auc:.6f}", flush=True)
    print(f"Full OOF Fold-Rank ROC-AUC   : {overall_oof_rank_auc:.6f}", flush=True)
    print(f"Unique Test Rank Values      : {len(np.unique(test_nn))} / {n_test}", flush=True)
    print("-" * 72, flush=True)

    np.savez_compressed(
        output_npz_path,
        oof_nn=oof_nn,
        oof_nn_rank=oof_nn_rank_norm,
        oof_nn_prob=oof_nn_prob.astype(np.float64),
        test_nn=test_nn,
        test_nn_prob=test_nn_prob.astype(np.float64),
        fold_aucs=fold_aucs,
        sk_id_curr_test=sk_id_curr_test,
        feature_names=feature_names
    )

    sub_df = pd.DataFrame({
        "SK_ID_CURR": sk_id_curr_test,
        "TARGET": test_nn
    })
    sub_df.to_csv(submission_csv_path, index=False, float_format="%.15f")
    print(f"Saved OOF & Test arrays to {output_npz_path} and {submission_csv_path}", flush=True)

if __name__ == "__main__":
    main()
