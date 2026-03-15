# ablation_08_wo_DTW.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from embedding import TrafficEmbedding
from DSGL import (
    MultiGranularityTemporalNet,
    DynamicMemorySpatialGraph,
    FusionAndOutput,
)
from FL import FLClient, FLAggregator, FederatedRunner

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


class StandardScaler:
    def __init__(self, eps=1e-6):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=0)
        std = x.std(dim=0)
        self.std = torch.clamp(std, min=self.eps)

    def transform(self, x: torch.Tensor):
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor):
        if self.mean is None or self.std is None:
            return x
        return x * self.std + self.mean


def masked_mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))


def masked_rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))


def masked_mape(preds, labels, thr=10.0):
    eps = 1e-6
    denom = torch.clamp(torch.abs(labels), min=thr)
    return (torch.abs(preds - labels) / (denom + eps)).mean() * 100


def split_data(data, train_ratio=0.7, val_ratio=0.1):
    T = data.shape[0]
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


def generate_windows(data, t_in=12, t_out=12):
    T, N, C = data.shape
    xs, ys = [], []
    for i in range(T - t_in - t_out + 1):
        xs.append(data[i:i + t_in])
        ys.append(data[i + t_in:i + t_in + t_out])
    return np.array(xs), np.array(ys)


def load_npz_with_ids(npz_path):
    z = np.load(npz_path)
    data = z["data"].astype(np.float32)
    node_order = None
    for key in ["index", "node_ids", "sensor_ids", "ids", "stations"]:
        if key in z:
            arr = z[key]
            try:
                node_order = [int(x) for x in np.array(arr).reshape(-1)]
            except Exception:
                node_order = [str(x) for x in np.array(arr).reshape(-1)]
            break
    return data, node_order


def build_adj_from_csv(csv_path, dataset_name, node_order=None):
    df = pd.read_csv(csv_path)
    assert "from" in df.columns and "to" in df.columns

    nameu = dataset_name.upper()
    if nameu in ["PEMS03", "PEMS3"] and "distance" in df.columns:
        dist = df["distance"].values.astype(float)
        col_used = "distance"
    elif "cost" in df.columns:
        dist = df["cost"].values.astype(float)
        col_used = "cost"
    elif "weight" in df.columns:
        dist = df["weight"].values.astype(float)
        col_used = "weight"
    else:
        dist = np.ones(len(df), dtype=float)
        col_used = "ones"
    print(f"[INFO] 邻接列使用: {col_used}")

    from_id, to_id = df["from"].values, df["to"].values
    uniq = sorted(set(from_id).union(set(to_id)))
    id2idx = {nid: i for i, nid in enumerate(uniq)}

    A = np.zeros((len(uniq), len(uniq)), dtype=np.float32)
    sigma = float(np.mean(dist)) if np.mean(dist) > 0 else 1.0
    for f, t, d in zip(from_id, to_id, dist):
        i, j = id2idx[f], id2idx[t]
        w = np.exp(- (float(d) / sigma) ** 2)
        A[i, j] = A[j, i] = w
    np.fill_diagonal(A, 1.0)

    if node_order is not None:
        remap = []
        miss = 0
        for nid in node_order:
            if nid in id2idx:
                remap.append(id2idx[nid])
            else:
                remap.append(-1)
                miss += 1
        if miss > 0:
            print(f"[WARN] {miss} nodes missing in CSV; use self-loop for them.")
        N = len(node_order)
        A_align = np.eye(N, dtype=np.float32)
        for i, ii in enumerate(remap):
            for j, jj in enumerate(remap):
                if ii >= 0 and jj >= 0:
                    A_align[i, j] = A[ii, jj]
        A = A_align

    deg = A.sum(1, keepdims=True) + 1e-6
    A = A / deg
    return A


# ---------- 简单语义注意（替代 SoftDTW） ----------
class SimpleSemanticAttention(nn.Module):
    """
    w/o DTW: 采用基于余弦相似度的简单语义图，显著弱于 SoftDTW。
    """
    def __init__(self, top_l: int = 5):
        super().__init__()
        self.top_l = top_l

    @staticmethod
    def _topl_mask(sim: torch.Tensor, l: int) -> torch.Tensor:
        B, N, _ = sim.shape
        sim = sim.clone()
        eye = torch.eye(N, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        sim = sim - 1e9 * eye
        idx = torch.topk(sim, k=l, dim=-1).indices
        M = torch.zeros_like(sim)
        b_ids = torch.arange(B, device=sim.device)[:, None, None].expand(B, N, l)
        n_ids = torch.arange(N, device=sim.device)[None, :, None].expand(B, N, l)
        M[b_ids, n_ids, idx] = 1.0
        return M

    def forward(self, H_temp: torch.Tensor, X_sp: torch.Tensor):
        B, T, N, d = H_temp.shape
        feat = H_temp.mean(dim=1)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        sim = torch.einsum("bnd,bmd->bnm", feat, feat)
        M = self._topl_mask(sim, self.top_l)
        A_sem = sim * M
        A_sem = A_sem / (A_sem.sum(-1, keepdim=True) + 1e-12)
        H_sem = torch.zeros_like(H_temp)  # 不额外加强表达，弱化语义分支
        return H_sem, A_sem


class STLearner_wo_DTW(nn.Module):
    """保留 MGTN + 动态空间，SoftDTW 换为简单语义注意。"""
    def __init__(self,
                 N: int,
                 d_model: int = 48,
                 H_future: int = 12,
                 out_dim: int = 1,
                 t_heads: int = 2,
                 t_k1: int = 3, t_dila1: int = 1,
                 t_k2: int = 3, t_dila2: int = 2,
                 mem_E: int = 32, mem_slots: int = 8, mem_heads: int = 2, node_embed_dim: int = 16,
                 diff_L: int = 2, dropout: float = 0.4,
                 sem_top_l: int = 5):
        super().__init__()
        self.temporal_net = MultiGranularityTemporalNet(
            d_model=d_model, n_heads=t_heads,
            k1=t_k1, dila1=t_dila1,
            k2=t_k2, dila2=t_dila2,
        )
        self.spatial_net = DynamicMemorySpatialGraph(
            d_model=d_model, E=mem_E, num_memory=mem_slots,
            n_heads=mem_heads, node_embed_dim=node_embed_dim,
            L=diff_L, dropout=dropout
        )
        self.semantic_net = SimpleSemanticAttention(top_l=sem_top_l)
        self.fusion_head = FusionAndOutput(d_model=d_model, z=H_future, out_dim=out_dim)
        self.N = N

    def forward(self, X_te: torch.Tensor, X_sp: torch.Tensor, A_static: torch.Tensor):
        H_temp = self.temporal_net(X_te)
        H_spatial, _, _ = self.spatial_net(H_temp, A_static=A_static)
        H_sem, _ = self.semantic_net(H_temp, X_sp)
        y_pred = self.fusion_head(H_temp, H_spatial, H_sem)
        return y_pred


class DSGFL_Wrapper_wo_DTW(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_channels=1,
                 d_model=48,
                 horizon=12):
        super().__init__()
        self.embed = TrafficEmbedding(in_channels=in_channels, d_model=d_model)
        self.st_learner = STLearner_wo_DTW(
            N=num_nodes,
            d_model=d_model,
            H_future=horizon,
            out_dim=1,
        )
        self.register_buffer("A_default", None)

    def set_default_adj(self, A: torch.Tensor):
        self.A_default = A

    def forward(self, X: torch.Tensor):
        if self.A_default is None:
            raise RuntimeError("Adjacency not set")
        X_te, X_sp = self.embed(X, self.A_default)
        y_pred = self.st_learner(X_te, X_sp, self.A_default)
        return y_pred


def run_pems08_wo_DTW(root, device="cuda", t_in=12, t_out=12):
    name_upper = "PEMS08"
    folder = os.path.join(root, name_upper)
    npz_path = os.path.join(folder, f"{name_upper}.npz")
    csv_path = os.path.join(folder, f"{name_upper}.csv")

    data_np, node_order = load_npz_with_ids(npz_path)
    print(f"[INFO] raw data shape = {data_np.shape}")
    num_nodes = data_np.shape[1]

    train_np, _, test_np = split_data(data_np)
    x_tr_np, y_tr_np = generate_windows(train_np, t_in, t_out)
    x_te_np, y_te_np = generate_windows(test_np, t_in, t_out)

    scaler = StandardScaler()
    scaler.fit(torch.from_numpy(train_np))

    x_tr = scaler.transform(torch.from_numpy(x_tr_np))
    y_tr = scaler.transform(torch.from_numpy(y_tr_np))
    x_te = scaler.transform(torch.from_numpy(x_te_np))
    y_te = scaler.transform(torch.from_numpy(y_te_np))

    A = build_adj_from_csv(csv_path, name_upper, node_order)
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    rounds = 8
    clients = 4
    batch = 16
    lr = 7e-4
    weight_decay = 0.0

    def make_loader(x, y):
        ds = TensorDataset(x.to(torch.float32), y.to(torch.float32))
        return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)

    idx_all = np.arange(len(x_tr))
    client_indices = np.array_split(idx_all, clients)
    train_loaders = [make_loader(x_tr[idx], y_tr[idx]) for idx in client_indices]

    def build_model():
        in_channels = data_np.shape[2]
        m = DSGFL_Wrapper_wo_DTW(
            num_nodes=num_nodes,
            in_channels=in_channels,
            d_model=48,
            horizon=t_out,
        ).to(device)
        m.set_default_adj(A_tensor)
        return m

    client_models = [build_model() for _ in range(clients)]
    global_model = build_model()

    def loss_on_norm(pred, yb):
        return torch.mean(torch.abs(pred - yb))

    def opt_ctor(params):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    fl_clients = [
        FLClient(
            client_id=i,
            model=client_models[i],
            loss_fn=loss_on_norm,
            optimizer_ctor=opt_ctor,
            device=torch.device(device),
        )
        for i in range(clients)
    ]

    aggregator = FLAggregator(sigma=8.0, tau=1000.0, device=torch.device(device))

    runner = FederatedRunner(
        server=aggregator,
        clients=fl_clients,
        tol=0.0,
        max_rounds=rounds,
        verbose=True,
    )

    def eta_schedule(step_idx: int) -> float:
        return float(1.2e-3 * (0.985 ** step_idx))

    print("\n=== Start FL Training: PEMS08 w/o DTW ===")
    runner.run(
        train_loaders=train_loaders,
        local_epochs=1,
        local_lr=lr,
        weight_decay=weight_decay,
        max_batches_per_client=None,
        stepwise_eta_prox=eta_schedule,
        ema_beta_for_sum=0.1,
    )

    global_model.load_state_dict(fl_clients[0].model.state_dict(), strict=False)
    global_model.eval()

    test_loader = DataLoader(
        TensorDataset(x_te, y_te),
        batch_size=64,
        shuffle=False,
    )

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                out = global_model(xb)
            out_denorm = scaler.inverse_transform(out.cpu())
            yb_denorm = scaler.inverse_transform(yb)
            preds.append(out_denorm)
            trues.append(yb_denorm)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    mae = masked_mae(pred, true).item()
    rmse = masked_rmse(pred, true).item()
    mape = masked_mape(pred, true).item()

    print(f"[Final] PEMS08 w/o DTW: MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")
    return mae, rmse, mape


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_pems08_wo_DTW(root=args.root, device=args.device)
