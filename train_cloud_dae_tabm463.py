import os
import sys
import copy
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler
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
    raise FileNotFoundError("Could not find 463-feature matrix in NPZ or Parquet format under data/, output/, or current dir.")

def extract_deduplicated_nan_signatures(X_tr: np.ndarray, X_te: np.ndarray):
    X_all = np.vstack([X_tr, X_te])
    nan_mat = np.isnan(X_all)
    nan_rate = nan_mat.mean(axis=0)
    active_cols = np.where((nan_rate >= 0.002) & (nan_rate <= 0.998))[0]
    row_nan_count = nan_mat.sum(axis=1, keepdims=True).astype(np.float32) / float(X_all.shape[1])
    if len(active_cols) == 0:
        return row_nan_count[:len(X_tr)], row_nan_count[len(X_tr):], active_cols
    sub_masks = nan_mat[:, active_cols].astype(np.float32)
    rng = np.random.RandomState(42)
    proj_sigs = (sub_masks.T @ rng.randn(sub_masks.shape[0], 5)).round(4)
    _, unique_idx = np.unique(proj_sigs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    dedup_masks = sub_masks[:, unique_idx]
    out_all = np.hstack([row_nan_count, dedup_masks]).astype(np.float32)
    return out_all[:len(X_tr)], out_all[len(X_tr):], active_cols[unique_idx]

class SwapNoiseDenoisingAutoencoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU()
        )
        self.enc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.10)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU()
        )
        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        self.dec2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.SiLU()
        )
        self.recon_head = nn.Linear(1024, in_dim)
        self.mask_head = nn.Linear(1024, in_dim)

    def forward(self, x: torch.Tensor):
        h0 = self.stem(x)
        h1 = self.enc1(h0)
        z = self.bottleneck(h1)
        d1 = self.dec1(z) + h1
        d2 = self.dec2(d1) + h0
        x_hat = self.recon_head(d2)
        c_logits = self.mask_head(d2)
        return x_hat, c_logits, z

def train_transductive_swap_dae(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    nan_sig_tr: np.ndarray,
    nan_sig_te: np.ndarray,
    device: torch.device,
    latent_dim: int = 128,
    swap_prob: float = 0.15,
    epochs: int = 24,
    batch_size: int = 1024,
    seed: int = 42
):
    set_all_seeds(seed)
    X_all = np.vstack([X_tr, X_te]).astype(np.float32)
    medians = np.nanmedian(X_all, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)
    X_imp = np.where(np.isnan(X_all), medians, X_all)
    qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal", subsample=120000, random_state=seed)
    X_qt = np.clip(qt.fit_transform(X_imp), -4.5, 4.5).astype(np.float32)
    nan_sigs = np.vstack([nan_sig_tr, nan_sig_te]).astype(np.float32)
    X_dae_in = np.hstack([X_qt, nan_sigs]).astype(np.float32)
    in_dim = X_dae_in.shape[1]
    n_total = X_dae_in.shape[0]
    model = SwapNoiseDenoisingAutoencoder(in_dim=in_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    ds = TensorDataset(torch.from_numpy(X_dae_in))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == "cuda"))
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Starting Swap-Noise DAE Pre-training ({epochs} epochs, batch_size={batch_size}, amp={amp_dtype})...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            bsz, fdim = xb.shape
            swap_mask = (torch.rand(bsz, fdim, device=device) < swap_prob).float()
            perm = torch.randint(0, bsz, (bsz, fdim), device=device)
            shuffled_xb = torch.gather(xb, 0, perm)
            corrupted_xb = (1.0 - swap_mask) * xb + swap_mask * shuffled_xb
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                x_hat, c_logits, _ = model(corrupted_xb)
                loss_recon = F.huber_loss(x_hat, xb, delta=1.0)
                loss_mask = F.binary_cross_entropy_with_logits(c_logits, swap_mask)
                loss = loss_recon + 0.40 * loss_mask
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        print(f"DAE Epoch {epoch:02d}/{epochs:02d} | Recon+Mask Loss: {total_loss / max(1, n_batches):.5f}", flush=True)
    model.eval()
    Z_list = []
    err_list = []
    X_tensor = torch.from_numpy(X_dae_in)
    with torch.no_grad():
        for start in range(0, n_total, 2048):
            xb = X_tensor[start:start + 2048].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                x_hat, c_logits, z = model(xb)
                diff = xb - x_hat
                l2_err = torch.mean(diff ** 2, dim=1, keepdim=True)
                l1_err = torch.mean(torch.abs(diff), dim=1, keepdim=True)
                max_err = torch.max(torch.abs(diff), dim=1, keepdim=True)[0]
                anom_prob = torch.sigmoid(c_logits).mean(axis=1, keepdim=True)
                errs = torch.cat([torch.log1p(l2_err), torch.log1p(l1_err), torch.log1p(max_err), anom_prob], dim=1)
            Z_list.append(z.float().cpu().numpy().astype(np.float32))
            err_list.append(errs.float().cpu().numpy().astype(np.float32))
    Z_all = np.vstack(Z_list)
    E_all = np.vstack(err_list)
    DAE_all = np.hstack([Z_all, E_all]).astype(np.float32)
    n_tr = len(X_tr)
    return DAE_all[:n_tr], DAE_all[n_tr:]

class LearnableNaNPeriodicEmbedding(nn.Module):
    def __init__(self, n_features: int, n_frequencies: int = 4, sigma: float = 0.05):
        super().__init__()
        self.n_features = n_features
        self.freqs = nn.Parameter(torch.randn(1, n_features, n_frequencies) * sigma)
        self.phases = nn.Parameter(torch.zeros(1, n_features, n_frequencies))
        self.proj_w = nn.Parameter(
            torch.randn(n_features, n_frequencies * 2, 3) * (1.0 / np.sqrt(n_frequencies * 2))
        )
        self.proj_b = nn.Parameter(torch.zeros(1, n_features, 3))
        self.nan_tokens = nn.Parameter(torch.randn(1, n_features, 4) * 0.02)
        self.act = nn.SiLU()

    def forward(self, x_val: torch.Tensor, x_isnan: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * np.pi * x_val.unsqueeze(-1) * self.freqs + self.phases
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        emb = self.act(torch.einsum("bfm,fmc->bfc", fourier, self.proj_w) + self.proj_b)
        obs_vec = torch.cat([x_val.unsqueeze(-1), emb], dim=-1)
        mask_exp = x_isnan.unsqueeze(-1)
        fused = (1.0 - mask_exp) * obs_vec + mask_exp * self.nan_tokens
        return fused.reshape(x_val.size(0), -1)

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

class TabM32RelationalDAE(nn.Module):
    def __init__(self, n_raw_features: int, n_aux_features: int, k: int = 32, d_model: int = 384, n_blocks: int = 3):
        super().__init__()
        self.n_raw_features = n_raw_features
        self.k = k
        self.nan_plr = LearnableNaNPeriodicEmbedding(n_raw_features, n_frequencies=4, sigma=0.05)
        total_in_dim = n_raw_features * 4 + n_aux_features
        self.input_norm = nn.LayerNorm(total_in_dim)
        self.input_drop = nn.Dropout(0.12)
        self.stem_be = LinearBatchEnsemble(total_in_dim, d_model, k=k, first_layer=True)
        self.stem_act = nn.SiLU()
        self.stem_drop = nn.Dropout(0.20)
        self.blocks = nn.ModuleList([
            TabMResidualBlock(d_model=d_model, k=k, dropout=0.22 + 0.03 * i)
            for i in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_model * 2)
        self.head = IndependentEnsembleReadout(d_model * 2, k=k)

    def forward(self, x_gauss: torch.Tensor, x_isnan: torch.Tensor, x_aux: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        if self.training and noise_std > 0.0:
            x_gauss = x_gauss + torch.randn_like(x_gauss) * noise_std * (1.0 - x_isnan)
        emb = self.nan_plr(x_gauss, x_isnan)
        full_in = self.input_drop(self.input_norm(torch.cat([emb, x_aux], dim=1)))
        xk = full_in.unsqueeze(1).expand(-1, self.k, -1)
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

def build_decoupled_optimizer(model: nn.Module, lr: float = 1.6e-3, weight_decay: float = 4e-4):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ["r_adapter", "s_adapter", "bias", "norm", "nan_tokens", "res_scale", "phases"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ],
        lr=lr
    )

def preprocess_fold_463(
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

@torch.no_grad()
def predict_tabm_463(
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
            logits_k = model(b_qt, b_nan, b_aux, noise_std=0.0)
            probs_k = torch.sigmoid(logits_k).float().mean(dim=1).detach().cpu().numpy().astype(np.float64)
        preds.append(probs_k)
    return np.concatenate(preds, axis=0)

def train_tabm_463_seed(
    tr_tuple: tuple,
    y_tr_fold: np.ndarray,
    va_tuple: tuple,
    y_va_fold: np.ndarray,
    te_tuple: tuple,
    device: torch.device,
    seed: int,
    k_heads: int = 32,
    max_epochs: int = 32,
    batch_size: int = 512,
    patience: int = 8,
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
    model = TabM32RelationalDAE(
        n_raw_features=n_raw,
        n_aux_features=n_aux,
        k=k_heads,
        d_model=384,
        n_blocks=3
    ).to(device)
    criterion = TabMIndependentLoss(pos_weight=2.0, smoothing=0.004)
    optimizer = build_decoupled_optimizer(model, lr=1.6e-3, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1.6e-3,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.12,
        div_factor=15.0,
        final_div_factor=100.0
    )
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
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
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits_k = model(b_qt, b_nan, b_aux, noise_std=0.012)
                loss = criterion(logits_k, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        va_preds = predict_tabm_463(model, X_va_qt, isnan_va, aux_va, batch_size=2048, device=device)
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
        ckpt_te_preds = predict_tabm_463(model, X_te_qt, isnan_te, aux_te, batch_size=2048, device=device)
        va_prob += w * ckpt_va_preds
        va_rank += w * to_average_uniform_rank_f64(ckpt_va_preds)
        te_prob += w * ckpt_te_preds
        te_rank += w * to_average_uniform_rank_f64(ckpt_te_preds)
    return va_rank, va_prob, te_rank, te_prob

def main():
    os.makedirs("output", exist_ok=True)
    output_npz_path = "output/oof_and_test_tabm32_463_dae.npz"
    dae_npz_path = "output/dae_128_latent_features.npz"
    submission_csv_path = "output/submission_tabm32_463_dae_rank.csv"
    n_splits = 10
    cv_split_seed = 42
    model_seeds = [707, 808]
    k_heads = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB", flush=True)
    X_tr, X_te, y_tr, sk_id_test, feature_cols = load_data()
    nan_sig_tr, nan_sig_te, _ = extract_deduplicated_nan_signatures(X_tr, X_te)
    print(f"Loaded 463 Relational Matrix: Train={X_tr.shape}, Test={X_te.shape} | Dedup NaN Sigs={nan_sig_tr.shape[1]}", flush=True)
    if os.path.exists(dae_npz_path):
        print(f"Loading cached DAE features from {dae_npz_path}...", flush=True)
        dae_loaded = np.load(dae_npz_path, allow_pickle=True)
        dae_tr = dae_loaded["dae_tr"]
        dae_te = dae_loaded["dae_te"]
    else:
        print("Training Unsupervised Swap-Noise DAE on combined 232,702 x 463 matrix...", flush=True)
        dae_tr, dae_te = train_transductive_swap_dae(
            X_tr=X_tr,
            X_te=X_te,
            nan_sig_tr=nan_sig_tr,
            nan_sig_te=nan_sig_te,
            device=device,
            latent_dim=128,
            swap_prob=0.15,
            epochs=24,
            batch_size=1024,
            seed=cv_split_seed
        )
        np.savez_compressed(
            dae_npz_path,
            dae_tr=dae_tr,
            dae_te=dae_te,
            sk_id_curr_test=sk_id_test
        )
        print(f"Saved 132-D DAE Latent & Anomaly Manifold to {dae_npz_path}", flush=True)
    n_train = X_tr.shape[0]
    n_test = X_te.shape[0]
    oof_rank = np.zeros(n_train, dtype=np.float64)
    oof_prob = np.zeros(n_train, dtype=np.float64)
    test_rank_accum = np.zeros(n_test, dtype=np.float64)
    test_prob_accum = np.zeros(n_test, dtype=np.float64)
    fold_aucs = np.zeros(n_splits, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_split_seed)
    print("Starting 10-Fold TabM-32 Training on 463 Relational + DAE Features...", flush=True)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr), start=1):
        t0_f = os.times().elapsed if hasattr(os.times(), "elapsed") else 0.0
        tr_tup, va_tup, te_tup = preprocess_fold_463(
            X_tr[tr_idx], X_tr[va_idx], X_te,
            dae_tr[tr_idx], dae_tr[va_idx], dae_te,
            nan_sig_tr[tr_idx], nan_sig_tr[va_idx], nan_sig_te,
            seed=cv_split_seed + fold_idx
        )
        y_tr_fold = y_tr[tr_idx]
        y_va_fold = y_tr[va_idx]
        f_va_r = np.zeros(len(va_idx), dtype=np.float64)
        f_va_p = np.zeros(len(va_idx), dtype=np.float64)
        f_te_r = np.zeros(n_test, dtype=np.float64)
        f_te_p = np.zeros(n_test, dtype=np.float64)
        for m_seed in model_seeds:
            vr, vp, tr_p, tp = train_tabm_463_seed(
                tr_tuple=tr_tup,
                y_tr_fold=y_tr_fold,
                va_tuple=va_tup,
                y_va_fold=y_va_fold,
                te_tuple=te_tup,
                device=device,
                seed=m_seed + fold_idx * 1000,
                k_heads=k_heads,
                max_epochs=32,
                batch_size=512,
                patience=8,
                top_k_ckpts=3
            )
            f_va_r += to_average_uniform_rank_f64(vr) / len(model_seeds)
            f_va_p += vp / len(model_seeds)
            f_te_r += to_average_uniform_rank_f64(tr_p) / len(model_seeds)
            f_te_p += tp / len(model_seeds)
        f_auc = float(roc_auc_score(y_va_fold, f_va_r))
        fold_aucs[fold_idx - 1] = f_auc
        oof_rank[va_idx] = to_average_uniform_rank_f64(f_va_r)
        oof_prob[va_idx] = f_va_p
        test_rank_accum += to_average_uniform_rank_f64(f_te_r) / n_splits
        test_prob_accum += f_te_p / n_splits
        print(f"Fold {fold_idx:02d}/{n_splits:02d} | TabM-32 (463+DAE) Validation ROC-AUC: {f_auc:.6f}", flush=True)
    full_oof_auc = float(roc_auc_score(y_tr, oof_rank))
    test_strict = to_strict_uniform_rank_f64(test_rank_accum, tie_breaker_arr=test_prob_accum)
    print("=" * 78, flush=True)
    print(f"TabM-32 (463 Relational + 132 DAE) Full OOF ROC-AUC : {full_oof_auc:.6f}", flush=True)
    print(f"Unique Test Predictions : {len(np.unique(test_strict))} / {n_test}", flush=True)
    print("=" * 78, flush=True)
    np.savez_compressed(
        output_npz_path,
        oof_tabm463_rank=oof_rank,
        oof_tabm463_prob=oof_prob,
        test_tabm463_rank=test_strict,
        test_tabm463_prob=test_prob_accum,
        fold_aucs=fold_aucs,
        sk_id_curr_test=sk_id_test,
        feature_names=feature_cols
    )
    sub_df = pd.DataFrame({
        "SK_ID_CURR": sk_id_test,
        "TARGET": test_strict
    })
    sub_df.to_csv(submission_csv_path, index=False, float_format="%.15f")
    print(f"Saved TabM-32 standalone submission to: {submission_csv_path}", flush=True)

if __name__ == "__main__":
    main()
