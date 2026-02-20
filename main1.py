# =========================== main1_tuned.py ===========================
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from embedding import TrafficEmbedding
from DSGL import STFeatureLearner
from FL import FLClient, FLAggregator, FederatedRunner

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ---------- Repro ----------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- StandardScaler ----------
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


# ---------- Metrics ----------
def masked_mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))

def masked_rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))

def masked_mape(preds, labels, thr=10.0):
    eps = 1e-6
    denom = torch.clamp(torch.abs(labels), min=thr)
    return (torch.abs(preds - labels) / (denom + eps)).mean() * 100


# ---------- Split & Windows ----------
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


# ---------- Load npz ----------
def load_npz_with_ids(npz_path):
    z = np.load(npz_path)
    # 保持与你现有 main1.py 一致：要求 key="data"
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


# ---------- Build adjacency (重要改动：不做行归一化，避免 DSGL 再对称归一化时“二次归一化”) ----------
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

    # 保留自环
    np.fill_diagonal(A, 1.0)

    # 对齐到 npz 的节点顺序
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
            if ii < 0:
                continue
            for j, jj in enumerate(remap):
                if jj < 0:
                    continue
                A_align[i, j] = A[ii, jj]
        A = A_align

    # ✅ 不再做 A = A/deg 行归一化
    return A


# ---------- DSGFL Wrapper ----------
class DSGFL_Wrapper(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_channels=1,
                 d_model=96,
                 horizon=12,
                 t_heads=4,
                 t_k1=3, t_dila1=1,
                 t_k2=3, t_dila2=2,
                 mem_E=64, mem_slots=16, mem_heads=4, node_embed_dim=16,
                 diff_L=3, dropout=0.10,
                 sem_top_l=4, sem_gamma=0.6, sem_window=4):
        super().__init__()
        self.embed = TrafficEmbedding(in_channels=in_channels, d_model=d_model)
        self.st_learner = STFeatureLearner(
            N=num_nodes,
            d_model=d_model,
            H_future=horizon,
            out_dim=1,
            t_heads=t_heads,
            t_k1=t_k1, t_dila1=t_dila1,
            t_k2=t_k2, t_dila2=t_dila2,
            mem_E=mem_E, mem_slots=mem_slots,
            mem_heads=mem_heads, node_embed_dim=node_embed_dim,
            diff_L=diff_L, dropout=dropout,
            sem_top_l=sem_top_l, sem_gamma=sem_gamma, sem_window=sem_window,
        )
        self.register_buffer("A_default", None)

    def set_default_adj(self, A: torch.Tensor):
        self.A_default = A

    def forward(self, X: torch.Tensor):
        if self.A_default is None:
            raise RuntimeError("Adjacency not set in DSGFL_Wrapper")
        X_te, X_sp = self.embed(X, self.A_default)
        y_pred, _ = self.st_learner(X_te, X_sp, self.A_default)
        return y_pred


# ---------- Eval ----------
@torch.no_grad()
def eval_model(model, scaler, x, y, device="cuda", batch_size=64):
    model.eval()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            out = model(xb)
        out_denorm = scaler.inverse_transform(out.cpu())
        yb_denorm = scaler.inverse_transform(yb)
        preds.append(out_denorm)
        trues.append(yb_denorm)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    mae = masked_mae(pred, true).item()
    rmse = masked_rmse(pred, true).item()
    mape = masked_mape(pred, true).item()
    return mae, rmse, mape


# ---------- Main runner ----------
def run_dataset(name, root, device="cuda", t_in=12, t_out=12, seed=42):
    set_seed(seed)

    name_upper = str(name).upper()
    print(f"\n=== Running dataset {name_upper} (horizon={t_out}) ===")
    folder = os.path.join(root, name_upper)
    npz_path = os.path.join(folder, f"{name_upper}.npz")
    csv_path = os.path.join(folder, f"{name_upper}.csv")

    # 1) load data
    data_np, node_order = load_npz_with_ids(npz_path)
    print(f"[INFO] raw data shape = {data_np.shape}")
    num_nodes = data_np.shape[1]
    in_channels = data_np.shape[2]

    # 2) split & windows (train/val/test)
    train_np, val_np, test_np = split_data(data_np)
    x_tr_np, y_tr_np = generate_windows(train_np, t_in, t_out)
    x_va_np, y_va_np = generate_windows(val_np, t_in, t_out)
    x_te_np, y_te_np = generate_windows(test_np, t_in, t_out)
    print(f"[INFO] x_train={x_tr_np.shape}, x_val={x_va_np.shape}, x_test={x_te_np.shape}")

    # 3) normalize (fit on train only)
    scaler = StandardScaler()
    scaler.fit(torch.from_numpy(train_np))
    x_tr = scaler.transform(torch.from_numpy(x_tr_np))
    y_tr = scaler.transform(torch.from_numpy(y_tr_np))
    x_va = scaler.transform(torch.from_numpy(x_va_np))
    y_va = scaler.transform(torch.from_numpy(y_va_np))
    x_te = scaler.transform(torch.from_numpy(x_te_np))
    y_te = scaler.transform(torch.from_numpy(y_te_np))

    # 4) adjacency (raw weighted A)
    A = build_adj_from_csv(csv_path, name_upper, node_order)
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    # 5) Federated hyperparams (更稳：降低 lr，适度 local_epochs，减小 proximal 强度)
    #    轻微按数据集规模分档（四个数据集都会更稳）
    rounds = 40
    clients = 4
    batch = 32
    local_epochs = 2
    lr = 2e-4
    weight_decay = 5e-4
    max_batches_per_client = None
    tol = 0.0

    # 大图（PEMS03/04）更容易震荡：更小 lr + 稍大 batch
    if name_upper in ["PEMS03", "PEMS3", "PEMS04", "PEMS4"]:
        rounds = 35
        clients = 4
        batch = 48
        local_epochs = 2
        lr = 1.5e-4
        weight_decay = 4e-4

    # 小图（PEMS07/08）适度加快
    if name_upper in ["PEMS07", "PEMS7", "PEMS08", "PEMS8"]:
        rounds = 28
        clients = 3
        batch = 24
        local_epochs = 2
        lr = 2.0e-4
        weight_decay = 5e-4
        max_batches_per_client = 80
        tol = 1e-4

    print(f"[FL-CONFIG] rounds={rounds}, clients={clients}, batch={batch}, "
          f"local_epochs={local_epochs}, lr={lr}, wd={weight_decay}, "
          f"max_batches={max_batches_per_client}, tol={tol}")

    # 6) split to clients (重要改动：shuffle 后再 split，降低 client 异质性)
    rng = np.random.RandomState(seed)
    idx_all = rng.permutation(len(x_tr))
    client_indices = np.array_split(idx_all, clients)

    def make_loader(x, y):
        ds = TensorDataset(x.to(torch.float32), y.to(torch.float32))
        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    train_loaders = [make_loader(x_tr[idx], y_tr[idx]) for idx in client_indices]

    # 7) model factory（轻微提升默认容量，跨数据集更稳）
    def build_model():
        m = DSGFL_Wrapper(
            num_nodes=num_nodes,
            in_channels=in_channels,
            d_model=96 if name_upper in ["PEMS03", "PEMS3", "PEMS04", "PEMS4"] else 88,
            mem_slots=16 if name_upper in ["PEMS03", "PEMS3", "PEMS04", "PEMS4"] else 14,
            diff_L=3,
            dropout=0.10,
            sem_top_l=4,
            sem_window=4,
            sem_gamma=0.6,
        ).to(device)
        m.set_default_adj(A_tensor)
        return m

    client_models = [build_model() for _ in range(clients)]

    # 8) loss & optimizer
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
            use_amp=True,
            grad_clip=2.0,     # 更稳一点
            verbose=False,
        )
        for i in range(clients)
    ]


    aggregator = FLAggregator(
        sigma=,
        tau=,
        device=torch.device(device),
    )

    runner = FederatedRunner(
        server=aggregator,
        clients=fl_clients,
        tol=tol,
        max_rounds=rounds,
        verbose=True,
    )

    # proximal schedule：更保守，避免破坏收敛（原来 1.2e-3 较激进）
    def eta_schedule(step_idx: int) -> float:
        return float(6e-4 * (0.992 ** step_idx))

    # 9) train + select best by val MAE
    print("\n=== Start Federated Training (select best by VAL) ===")
    best_state = None
    best_val_mae = float("inf")

    for r in range(1, rounds + 1):
        # 让 runner 每次只跑 1 round（复用其逻辑）
        runner.max_rounds = 1
        runner.run(
            train_loaders=train_loaders,
            local_epochs=local_epochs,
            local_lr=lr,
            weight_decay=weight_decay,
            max_batches_per_client=max_batches_per_client,
            stepwise_eta_prox=eta_schedule,
            ema_beta_for_sum=0.1,
        )

        # 取 client0 作为当前“全局快照”
        cur_model = fl_clients[0].model
        v_mae, v_rmse, v_mape = eval_model(cur_model, scaler, x_va, y_va, device=device, batch_size=64)
        print(f"[VAL] round={r:02d} MAE={v_mae:.4f} RMSE={v_rmse:.4f} MAPE={v_mape:.2f}%")

        if v_mae < best_val_mae:
            best_val_mae = v_mae
            best_state = copy.deepcopy(cur_model.state_dict())

    # 10) test with best
    fl_clients[0].model.load_state_dict(best_state, strict=False)
    t_mae, t_rmse, t_mape = eval_model(fl_clients[0].model, scaler, x_te, y_te, device=device, batch_size=64)
    print(f"\n[Final-BEST] {name_upper}: MAE={t_mae:.4f} RMSE={t_rmse:.4f} MAPE={t_mape:.2f}%")
    return t_mae, t_rmse, t_mape


# ---------- main ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mae, rmse, mape = run_dataset(args.dataset, root=args.root, device=args.device, seed=args.seed)

    print("\n=== Summary ===")
    print(f"{args.dataset.upper()}: MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")