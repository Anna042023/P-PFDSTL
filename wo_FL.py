# ablation_08_wo_FL.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from embedding import TrafficEmbedding
from DSGL import STFeatureLearner

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


class CentralWrapper(nn.Module):
    """w/o FL: 单模型中心化训练，保留完整 STFeatureLearner."""
    def __init__(self,
                 num_nodes,
                 in_channels=1,
                 d_model=48,
                 horizon=12):
        super().__init__()
        self.embed = TrafficEmbedding(in_channels=in_channels, d_model=d_model)
        self.st_learner = STFeatureLearner(
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
        y_pred, _ = self.st_learner(X_te, X_sp, self.A_default)
        return y_pred


def run_pems08_wo_FL(root, device="cuda", t_in=12, t_out=12,
                     epochs=8, batch_size=16, lr=7e-4, weight_decay=0.0,
                     patience=3):
    name_upper = "PEMS08"
    folder = os.path.join(root, name_upper)
    npz_path = os.path.join(folder, f"{name_upper}.npz")
    csv_path = os.path.join(folder, f"{name_upper}.csv")

    data_np, node_order = load_npz_with_ids(npz_path)
    print(f"[INFO] raw data shape = {data_np.shape}")
    num_nodes = data_np.shape[1]
    in_channels = data_np.shape[2]

    train_np, val_np, test_np = split_data(data_np)
    x_tr_np, y_tr_np = generate_windows(train_np, t_in, t_out)
    x_val_np, y_val_np = generate_windows(val_np, t_in, t_out)
    x_te_np, y_te_np = generate_windows(test_np, t_in, t_out)

    scaler = StandardScaler()
    scaler.fit(torch.from_numpy(train_np))

    x_tr = scaler.transform(torch.from_numpy(x_tr_np))
    y_tr = scaler.transform(torch.from_numpy(y_tr_np))
    x_val = scaler.transform(torch.from_numpy(x_val_np))
    y_val = scaler.transform(torch.from_numpy(y_val_np))
    x_te = scaler.transform(torch.from_numpy(x_te_np))
    y_te = scaler.transform(torch.from_numpy(y_te_np))

    A = build_adj_from_csv(csv_path, name_upper, node_order)
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    model = CentralWrapper(
        num_nodes=num_nodes,
        in_channels=in_channels,
        d_model=48,
        horizon=t_out,
    ).to(device)
    model.set_default_adj(A_tensor)

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_te, y_te),
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=True)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    def eval_on(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    out = model(xb)
                out_denorm = scaler.inverse_transform(out.cpu())
                yb_denorm = scaler.inverse_transform(yb.cpu())
                preds.append(out_denorm)
                trues.append(yb_denorm)
        pred = torch.cat(preds, dim=0)
        true = torch.cat(trues, dim=0)
        return masked_mae(pred, true).item()

    print("\n=== Start Centralized Training: PEMS08 w/o FL ===")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
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
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        val_mae = eval_on(val_loader)
        print(f"[Epoch {epoch}] train_L1={epoch_loss:.4f}  val_MAE={val_mae:.4f}")

        if val_mae + 1e-4 < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # Test
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                out = model(xb)
            out_denorm = scaler.inverse_transform(out.cpu())
            yb_denorm = scaler.inverse_transform(yb.cpu())
            preds.append(out_denorm)
            trues.append(yb_denorm)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    mae = masked_mae(pred, true).item()
    rmse = masked_rmse(pred, true).item()
    mape = masked_mape(pred, true).item()
    print(f"[Final] PEMS08 w/o FL (centralized): MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")
    return mae, rmse, mape


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_pems08_wo_FL(root=args.root, device=args.device)
