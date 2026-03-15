# sensitivity_chebK.py
# ------------------------------------------------------------
# Sensitivity analysis for Chebyshev polynomial order K in spatial embedding:
#   K in {2,3,4,5} -> test MAE/RMSE (best checkpoint selected by VAL MAE)
# ------------------------------------------------------------
import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from embedding import TrafficEmbedding  # cheb_K lives here
from DSGL import STFeatureLearner
from FL import FLClient, FLAggregator, FederatedRunner


torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ---------------- Repro ----------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- StandardScaler (same style as main1.py) ----------------
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


# ---------------- Metrics ----------------
def masked_mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))

def masked_rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))


# ---------------- Split & Windows ----------------
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


# ---------------- Load npz ----------------
def load_npz_with_ids(npz_path):
    z = np.load(npz_path)
    data = z["data"].astype(np.float32)  # keep consistent with main1.py
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


# ---------------- Build adjacency from csv (same logic as main1.py) ----------------
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
    print(f"[INFO] adjacency column used: {col_used}")

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

    # align to node order in npz
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

    return A


# ---------------- Model wrapper with configurable cheb_K ----------------
class DSGFL_Wrapper(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels=1,
        d_model=96,
        horizon=12,
        cheb_K=3,              # <- sensitivity variable
        t_heads=4,
        t_k1=3, t_dila1=1,
        t_k2=3, t_dila2=2,
        mem_E=64, mem_slots=16, mem_heads=4, node_embed_dim=16,
        diff_L=3, dropout=0.10,
        sem_top_l=4, sem_gamma=0.6, sem_window=4
    ):
        super().__init__()
        self.embed = TrafficEmbedding(in_channels=in_channels, d_model=d_model, cheb_K=cheb_K)
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


# ---------------- Eval ----------------
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
    return mae, rmse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="PEMS04")
    p.add_argument("--root", type=str, required=True, help="root folder that contains PEMSxx/PEMSxx.npz and PEMSxx/PEMSxx.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # windows
    p.add_argument("--t_in", type=int, default=12)
    p.add_argument("--t_out", type=int, default=12)

    # FL configs
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--max_batches_per_client", type=int, default=80)
    p.add_argument("--tol", type=float, default=0.0)

    # aggregator
    p.add_argument("--sigma", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=3000.0)

    # model capacity
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--mem_slots", type=int, default=16)
    p.add_argument("--diff_L", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--sem_top_l", type=int, default=4)
    p.add_argument("--sem_window", type=int, default=4)
    p.add_argument("--sem_gamma", type=float, default=0.6)

    # sensitivity Ks
    p.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4, 5])

    # outputs
    p.add_argument("--out_dir", type=str, default="sens_outputs")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    name_upper = args.dataset.upper()
    folder = os.path.join(args.root, name_upper)
    npz_path = os.path.join(folder, f"{name_upper}.npz")
    csv_path = os.path.join(folder, f"{name_upper}.csv")
    assert os.path.exists(npz_path), f"Missing: {npz_path}"
    assert os.path.exists(csv_path), f"Missing: {csv_path}"

    # 1) load data
    data_np, node_order = load_npz_with_ids(npz_path)
    num_nodes = data_np.shape[1]
    in_channels = data_np.shape[2]
    print(f"[INFO] data shape = {data_np.shape}, nodes={num_nodes}, channels={in_channels}")

    # 2) split & windows
    train_np, val_np, test_np = split_data(data_np)
    x_tr_np, y_tr_np = generate_windows(train_np, args.t_in, args.t_out)
    x_va_np, y_va_np = generate_windows(val_np, args.t_in, args.t_out)
    x_te_np, y_te_np = generate_windows(test_np, args.t_in, args.t_out)
    print(f"[INFO] x_train={x_tr_np.shape}, x_val={x_va_np.shape}, x_test={x_te_np.shape}")

    # 3) normalize (fit on train)
    scaler = StandardScaler()
    scaler.fit(torch.from_numpy(train_np))
    x_tr = scaler.transform(torch.from_numpy(x_tr_np))
    y_tr = scaler.transform(torch.from_numpy(y_tr_np))
    x_va = scaler.transform(torch.from_numpy(x_va_np))
    y_va = scaler.transform(torch.from_numpy(y_va_np))
    x_te = scaler.transform(torch.from_numpy(x_te_np))
    y_te = scaler.transform(torch.from_numpy(y_te_np))

    # 4) adjacency
    A = build_adj_from_csv(csv_path, name_upper, node_order=node_order)
    A_tensor = torch.tensor(A, dtype=torch.float32, device=args.device)

    # 5) split to clients (shuffle then split)
    rng = np.random.RandomState(args.seed)
    idx_all = rng.permutation(len(x_tr))
    client_indices = np.array_split(idx_all, args.clients)

    def make_loader(x, y):
        ds = TensorDataset(x.to(torch.float32), y.to(torch.float32))
        return DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    train_loaders = [make_loader(x_tr[idx], y_tr[idx]) for idx in client_indices]

    # 6) loss & optimizer
    def loss_on_norm(pred, yb):
        return torch.mean(torch.abs(pred - yb))

    def opt_ctor(params):
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # proximal schedule (keep close to your main1 settings)
    def eta_schedule(step_idx: int) -> float:
        return float(6e-4 * (0.992 ** step_idx))

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for K in args.Ks:
        print("\n" + "=" * 80)
        print(f"[SENS-K] Running K={K}")
        print("=" * 80)

        # build clients
        def build_model():
            m = DSGFL_Wrapper(
                num_nodes=num_nodes,
                in_channels=in_channels,
                d_model=args.d_model,
                horizon=args.t_out,
                cheb_K=K,  # <- HERE
                mem_slots=args.mem_slots,
                diff_L=args.diff_L,
                dropout=args.dropout,
                sem_top_l=args.sem_top_l,
                sem_window=args.sem_window,
                sem_gamma=args.sem_gamma,
            ).to(args.device)
            m.set_default_adj(A_tensor)
            return m

        client_models = [build_model() for _ in range(args.clients)]
        fl_clients = [
            FLClient(
                client_id=i,
                model=client_models[i],
                loss_fn=loss_on_norm,
                optimizer_ctor=opt_ctor,
                device=torch.device(args.device),
                use_amp=True,
                grad_clip=2.0,
                verbose=False,
            )
            for i in range(args.clients)
        ]

        aggregator = FLAggregator(sigma=args.sigma, tau=args.tau, device=torch.device(args.device))
        runner = FederatedRunner(server=aggregator, clients=fl_clients, tol=args.tol, max_rounds=args.rounds, verbose=False)

        # train: select best by VAL MAE (same pattern as your main1.py)
        best_state = None
        best_val_mae = float("inf")

        for r in range(1, args.rounds + 1):
            runner.max_rounds = 1
            runner.run(
                train_loaders=train_loaders,
                local_epochs=args.local_epochs,
                local_lr=args.lr,
                weight_decay=args.weight_decay,
                max_batches_per_client=args.max_batches_per_client,
                stepwise_eta_prox=eta_schedule,
                ema_beta_for_sum=0.1,
            )
            cur_model = fl_clients[0].model
            v_mae, v_rmse = eval_model(cur_model, scaler, x_va, y_va, device=args.device, batch_size=64)
            print(f"[K={K}] VAL round={r:02d}  MAE={v_mae:.4f} RMSE={v_rmse:.4f}")

            if v_mae < best_val_mae:
                best_val_mae = v_mae
                best_state = copy.deepcopy(cur_model.state_dict())

        # test best
        fl_clients[0].model.load_state_dict(best_state, strict=False)
        t_mae, t_rmse = eval_model(fl_clients[0].model, scaler, x_te, y_te, device=args.device, batch_size=64)
        print(f"[K={K}] TEST(best@val): MAE={t_mae:.4f} RMSE={t_rmse:.4f}")

        rows.append({"dataset": name_upper, "K": K, "test_mae": t_mae, "test_rmse": t_rmse})

    df = pd.DataFrame(rows).sort_values("K")
    out_csv = os.path.join(args.out_dir, f"sens_chebK_{name_upper}.csv")
    df.to_csv(out_csv, index=False)
    print("\n[SAVED]", out_csv)
    print(df)

    # optional plot
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(df["K"].values, df["test_mae"].values, marker="o")
        plt.xlabel("Chebyshev polynomial order K")
        plt.ylabel("Test MAE")
        plt.grid(True, alpha=0.3)
        out_png = os.path.join(args.out_dir, f"sens_chebK_{name_upper}_MAE.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.plot(df["K"].values, df["test_rmse"].values, marker="o")
        plt.xlabel("Chebyshev polynomial order K")
        plt.ylabel("Test RMSE")
        plt.grid(True, alpha=0.3)
        out_png2 = os.path.join(args.out_dir, f"sens_chebK_{name_upper}_RMSE.png")
        plt.savefig(out_png2, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print("[SAVED]", out_png)
        print("[SAVED]", out_png2)
    except Exception as e:
        print("[WARN] plot skipped:", repr(e))


if __name__ == "__main__":
    main()