# ablation_all.py
import os
import numpy as np
import torch
import torch.nn as nn

from embedding import TrafficEmbedding
from DSGL import (
    STFeatureLearner,
    MultiGranularityTemporalNet,
    GlobalTemporalAttention,
    LocalTemporalCausalConv,
    AdaptiveTemporalFusion,
    DynamicMemorySpatialGraph,
    SoftDTWSemanticAttention,
    FusionAndOutput,
    normalize_symmetric_adj,
    DiffusionSpatialAggregator,
)
from FL import FLClient, FLAggregator, FederatedRunner

from ablation_common import (
    StandardScaler, split_data, generate_windows,
    resolve_dataset_files, load_npz_with_ids, build_adj_from_csv,
    masked_mae, masked_rmse, masked_mape,
    make_loader
)


# ---------------- Temporal variants ----------------
class SingleGranularityTemporalNet(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.att = GlobalTemporalAttention(d_model, n_heads=n_heads)
        self.conv = LocalTemporalCausalConv(d_model, kernel_size=kernel_size, dilation=dilation)
        self.fuse = AdaptiveTemporalFusion(d_model)

    def forward(self, X_te):
        return self.fuse(self.att(X_te), self.conv(X_te))

class TemporalNet_wo_TA(nn.Module):
    def __init__(self, d_model: int, k: int = 3, dilation: int = 1):
        super().__init__()
        self.conv = LocalTemporalCausalConv(d_model, kernel_size=k, dilation=dilation)
    def forward(self, X_te):
        return self.conv(X_te)

class TemporalNet_wo_TC(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.att = GlobalTemporalAttention(d_model, n_heads=n_heads)
    def forward(self, X_te):
        return self.att(X_te)


# ---------------- Spatial variants ----------------
class StaticSpatialGraph(nn.Module):
    """w/o DSGL: 静态扩散"""
    def __init__(self, d_model: int = 48, L: int = 2, dropout: float = 0.4):
        super().__init__()
        self.agg = DiffusionSpatialAggregator(d_model=d_model, L=L, dropout=dropout)
    def forward(self, H_temp, A_static):
        B, T, N, d = H_temp.shape
        A_hat = normalize_symmetric_adj(A_static.to(H_temp.device))
        A_hat_b = A_hat.unsqueeze(0).expand(B, -1, -1).contiguous()
        H_spatial = self.agg(H_temp, A_hat_b)
        return H_spatial, A_hat_b

class SpatialNet_fixed_static(nn.Module):
    """
    w/o AGC / w/o DGA：固定用 normalize(A_static)，返回 (H_spatial, A_dynamic=0, A_fused=A_hat)
    """
    def __init__(self, d_model: int = 48, L: int = 2, dropout: float = 0.4):
        super().__init__()
        self.agg = DiffusionSpatialAggregator(d_model=d_model, L=L, dropout=dropout)
    def forward(self, H_temp, A_static):
        B, T, N, d = H_temp.shape
        A_hat = normalize_symmetric_adj(A_static.to(H_temp.device))
        A_hat_b = A_hat.unsqueeze(0).expand(B, -1, -1).contiguous()
        H_spatial = self.agg(H_temp, A_hat_b)
        A_dynamic = torch.zeros_like(A_hat_b)
        A_fused = A_hat_b
        return H_spatial, A_dynamic, A_fused


# ---------------- Semantic variants ----------------
class SimpleSemanticAttention(nn.Module):
    """w/o DTW: 用余弦替代 SoftDTW，并且不提供语义特征更新（H_sem=0）"""
    def __init__(self, top_l=5):
        super().__init__()
        self.top_l = top_l

    @staticmethod
    def _topl_mask(sim, l):
        B, N, _ = sim.shape
        sim = sim.clone()
        eye = torch.eye(N, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        sim = sim - 1e9 * eye
        idx = torch.topk(sim, k=l, dim=-1).indices
        M = torch.zeros_like(sim)
        b = torch.arange(B, device=sim.device)[:, None, None].expand(B, N, l)
        n = torch.arange(N, device=sim.device)[None, :, None].expand(B, N, l)
        M[b, n, idx] = 1.0
        return M

    def forward(self, H_temp, X_sp):
        B, T, N, d = H_temp.shape
        feat = H_temp.mean(dim=1)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        sim = torch.einsum("bnd,bmd->bnm", feat, feat)
        M = self._topl_mask(sim, self.top_l)
        A_sem = sim * M
        A_sem = A_sem / (A_sem.sum(-1, keepdim=True) + 1e-12)
        H_sem = torch.zeros_like(H_temp)
        return H_sem, A_sem

class NoSemanticAttention(nn.Module):
    """w/o DSA: 完全移除语义分支（不构图）"""
    def forward(self, H_temp, X_sp):
        B, T, N, d = H_temp.shape
        H_sem = torch.zeros_like(H_temp)
        A_sem = torch.eye(N, device=H_temp.device, dtype=H_temp.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
        return H_sem, A_sem


# ---------------- Model builders per mode ----------------
class Wrapper(nn.Module):
    def __init__(self, core, num_nodes, in_channels, d_model, horizon):
        super().__init__()
        self.embed = TrafficEmbedding(in_channels=in_channels, d_model=d_model)
        self.core = core
        self.register_buffer("A_default", None)
        self.horizon = horizon

    def set_default_adj(self, A):
        self.A_default = A

    def forward(self, X):
        if self.A_default is None:
            raise RuntimeError("Adjacency not set")
        X_te, X_sp = self.embed(X, self.A_default)
        out = self.core(X_te, X_sp, self.A_default)
        # core 有的返回 y_pred，有的返回 (y_pred, aux)
        return out[0] if isinstance(out, (tuple, list)) else out


def build_core(mode: str, N: int, d_model: int, horizon: int,
               t_heads=2, t_k1=3, t_dila1=1, t_k2=3, t_dila2=2,
               mem_E=32, mem_slots=8, mem_heads=2, node_embed_dim=16,
               diff_L=2, dropout=0.4,
               sem_top_l=5, sem_gamma=0.5, sem_window=4):

    mode = mode.lower()

    if mode == "full":
        # 用你原始实现
        return STFeatureLearner(N=N, d_model=d_model, H_future=horizon, out_dim=1,
                               t_heads=t_heads, t_k1=t_k1, t_dila1=t_dila1, t_k2=t_k2, t_dila2=t_dila2,
                               mem_E=mem_E, mem_slots=mem_slots, mem_heads=mem_heads, node_embed_dim=node_embed_dim,
                               diff_L=diff_L, dropout=dropout,
                               sem_top_l=sem_top_l, sem_gamma=sem_gamma, sem_window=sem_window)

    # 下面用 “拆出来的三分支 + Fusion” 来保证每个消融可控
    if mode == "wo_mgtn":
        temporal = SingleGranularityTemporalNet(d_model, n_heads=t_heads, kernel_size=t_k1, dilation=t_dila1)
    elif mode == "wo_ta":
        temporal = TemporalNet_wo_TA(d_model, k=t_k1, dilation=t_dila1)
    elif mode == "wo_tc":
        temporal = TemporalNet_wo_TC(d_model, n_heads=t_heads)
    else:
        temporal = MultiGranularityTemporalNet(d_model=d_model, n_heads=t_heads, k1=t_k1, dila1=t_dila1, k2=t_k2, dila2=t_dila2)

    if mode == "wo_dsgl":
        spatial = StaticSpatialGraph(d_model=d_model, L=diff_L, dropout=dropout)
        spatial_kind = "static2"
    elif mode in ["wo_agc", "wo_dga"]:
        spatial = SpatialNet_fixed_static(d_model=d_model, L=diff_L, dropout=dropout)
        spatial_kind = "fixed3"
    else:
        spatial = DynamicMemorySpatialGraph(d_model=d_model, E=mem_E, num_memory=mem_slots,
                                            n_heads=mem_heads, node_embed_dim=node_embed_dim,
                                            L=diff_L, dropout=dropout)
        spatial_kind = "dynamic3"

    if mode == "wo_dtw":
        semantic = SimpleSemanticAttention(top_l=sem_top_l)
    elif mode == "wo_dsa":
        semantic = NoSemanticAttention()
    else:
        semantic = SoftDTWSemanticAttention(d_model=d_model, top_l=sem_top_l, gamma=sem_gamma, window=sem_window)

    fusion = FusionAndOutput(d_model=d_model, z=horizon, out_dim=1)

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = temporal
            self.spatial = spatial
            self.semantic = semantic
            self.fusion = fusion
            self.spatial_kind = spatial_kind

        def forward(self, X_te, X_sp, A_static):
            H_temp = self.temporal(X_te)

            if self.spatial_kind == "static2":
                H_sp, _ = self.spatial(H_temp, A_static)
            elif self.spatial_kind == "fixed3":
                H_sp, _, _ = self.spatial(H_temp, A_static)
            else:
                H_sp, _, _ = self.spatial(H_temp, A_static=A_static)

            H_sem, _ = self.semantic(H_temp, X_sp)
            y = self.fusion(H_temp, H_sp, H_sem)
            return y

    return Core()


# ---------------- Training & Eval ----------------
def eval_denorm(model, scaler, x, y, device="cuda", batch_size=64):
    model.eval()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                out = model(xb)
            preds.append(scaler.inverse_transform(out.cpu()))
            trues.append(scaler.inverse_transform(yb))
    pred = torch.cat(preds, 0)
    true = torch.cat(trues, 0)
    return (
        masked_mae(pred, true).item(),
        masked_rmse(pred, true).item(),
        masked_mape(pred, true).item()
    )


def train_central(model, train_loader, val_loader, scaler, device="cuda",
                  epochs=8, lr=7e-4, weight_decay=0.0, patience=3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=True)

    best = 1e18
    best_state = None
    bad = 0

    def val_mae():
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    out = model(xb)
                preds.append(scaler.inverse_transform(out.cpu()))
                trues.append(scaler.inverse_transform(yb))
        p = torch.cat(preds, 0)
        t = torch.cat(trues, 0)
        return masked_mae(p, t).item()

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                out = model(xb)
                loss = torch.mean(torch.abs(out - yb))
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()

        v = val_mae()
        print(f"[Epoch {ep}] val_MAE={v:.4f}")
        if v + 1e-4 < best:
            best = v
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)


def run(args):
    device = args.device
    folder, npz_path, csv_path, ds_upper = resolve_dataset_files(args.root, args.dataset)
    print(f"[INFO] dataset={ds_upper}")
    print(f"[INFO] npz={npz_path}")
    print(f"[INFO] csv={csv_path}")

    data_np, node_order = load_npz_with_ids(npz_path)
    num_nodes = data_np.shape[1]
    in_channels = data_np.shape[2]

    train_np, val_np, test_np = split_data(data_np)
    x_tr_np, y_tr_np = generate_windows(train_np, args.t_in, args.t_out)
    x_val_np, y_val_np = generate_windows(val_np, args.t_in, args.t_out)
    x_te_np, y_te_np = generate_windows(test_np, args.t_in, args.t_out)

    scaler = StandardScaler()
    scaler.fit(torch.from_numpy(train_np))

    x_tr = scaler.transform(torch.from_numpy(x_tr_np))
    y_tr = scaler.transform(torch.from_numpy(y_tr_np))
    x_val = scaler.transform(torch.from_numpy(x_val_np))
    y_val = scaler.transform(torch.from_numpy(y_val_np))
    x_te = scaler.transform(torch.from_numpy(x_te_np))
    y_te = scaler.transform(torch.from_numpy(y_te_np))

    A = build_adj_from_csv(csv_path, ds_upper, node_order)
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    mode = args.mode.lower()

    # ----- w/o FL: centralized -----
    if mode == "wo_fl":
        core = build_core("full", N=num_nodes, d_model=args.d_model, horizon=args.t_out)
        model = Wrapper(core, num_nodes, in_channels, args.d_model, args.t_out).to(device)
        model.set_default_adj(A_tensor)

        tr_loader = make_loader(x_tr, y_tr, batch_size=args.batch, shuffle=True)
        va_loader = make_loader(x_val, y_val, batch_size=args.batch, shuffle=False)

        print(f"\n=== Central Training ({ds_upper}) mode=wo_FL ===")
        train_central(model, tr_loader, va_loader, scaler,
                      device=device, epochs=args.epochs, lr=args.lr, weight_decay=args.wd, patience=args.patience)

        mae, rmse, mape = eval_denorm(model, scaler, x_te, y_te, device=device, batch_size=64)
        print(f"[Final] {ds_upper} wo_FL: MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")
        return

    # ----- FL modes -----
    core = build_core(mode, N=num_nodes, d_model=args.d_model, horizon=args.t_out)
    def build_model():
        m = Wrapper(core=build_core(mode, N=num_nodes, d_model=args.d_model, horizon=args.t_out),
                    num_nodes=num_nodes, in_channels=in_channels, d_model=args.d_model, horizon=args.t_out).to(device)
        m.set_default_adj(A_tensor)
        return m

    idx_all = np.arange(len(x_tr))
    client_indices = np.array_split(idx_all, args.clients)
    train_loaders = [make_loader(x_tr[idx], y_tr[idx], batch_size=args.batch, shuffle=True) for idx in client_indices]

    client_models = [build_model() for _ in range(args.clients)]

    def loss_on_norm(pred, yb):
        return torch.mean(torch.abs(pred - yb))

    def opt_ctor(params):
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    fl_clients = [
        FLClient(client_id=i, model=client_models[i], loss_fn=loss_on_norm, optimizer_ctor=opt_ctor, device=torch.device(device))
        for i in range(args.clients)
    ]

    aggregator = FLAggregator(sigma=args.sigma, tau=args.tau, device=torch.device(device))
    runner = FederatedRunner(server=aggregator, clients=fl_clients, tol=0.0, max_rounds=args.rounds, verbose=True)

    def eta_schedule(step_idx: int) -> float:
        return float(args.eta0 * (args.eta_decay ** step_idx))

    print(f"\n=== FL Training ({ds_upper}) mode={args.mode} ===")
    runner.run(
        train_loaders=train_loaders,
        local_epochs=args.local_epochs,
        local_lr=args.lr,
        weight_decay=args.wd,
        max_batches_per_client=None,
        stepwise_eta_prox=eta_schedule,
        ema_beta_for_sum=args.ema_beta,
    )

    model = fl_clients[0].model
    mae, rmse, mape = eval_denorm(model, scaler, x_te, y_te, device=device, batch_size=64)
    print(f"[Final] {ds_upper} {args.mode}: MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, help="PEMS03/PEMS04/PEMS07/PEMS08")
    p.add_argument("--mode", type=str, default="full",
                   help="full|wo_MGTN|wo_DSGL|wo_DTW|wo_FL|wo_TA|wo_TC|wo_AGC|wo_DGA|wo_DSA")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--t_in", type=int, default=12)
    p.add_argument("--t_out", type=int, default=12)
    p.add_argument("--d_model", type=int, default=48)

    # FL args
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)

    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--tau", type=float, default=1000.0)
    p.add_argument("--eta0", type=float, default=1.2e-3)
    p.add_argument("--eta_decay", type=float, default=0.985)
    p.add_argument("--ema_beta", type=float, default=0.1)

    # central args
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)

    args = p.parse_args()
    run(args)