# sensitivity_comm_rounds.py
# ------------------------------------------------------------
# Sensitivity analysis for communication rounds:
#   record avg_param_delta per round:
#     Delta_avg^(k) = (1/N) * sum_i ||theta_i^(k) - theta_i^(k-1)||_2
#   report at rounds {10,15,20,25,30,35,40} (paper Fig.4 style)
# ------------------------------------------------------------
import os
import argparse
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


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# ----- flatten & avg delta (same idea as FederatedRunner._avg_param_delta) -----
def _flat_params(state_dict):
    vecs = []
    for _, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            vecs.append(v.detach().reshape(-1).cpu().float())
    return torch.cat(vecs) if vecs else torch.empty(0)

@torch.no_grad()
def avg_param_delta(prev_states, cur_states):
    vals = []
    for sp, sc in zip(prev_states, cur_states):
        vp, vc = _flat_params(sp), _flat_params(sc)
        L = min(vp.numel(), vc.numel())
        if L > 0:
            vals.append(torch.norm(vp[:L] - vc[:L], p=2).item())
    return float(sum(vals) / max(1, len(vals)))


class DSGFL_Wrapper(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels=1,
        d_model=96,
        horizon=12,
        cheb_K=3,
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="PEMS04")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--t_in", type=int, default=12)
    p.add_argument("--t_out", type=int, default=12)

    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--max_batches_per_client", type=int, default=80)

    p.add_argument("--sigma", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=3000.0)

    # keep K fixed here (Fig.4 studies rounds; paper says K=4 tends to be best)
    p.add_argument("--cheb_K", type=int, default=4)

    # which rounds to report
    p.add_argument("--report_rounds", type=int, nargs="+", default=[10, 15, 20, 25, 30, 35, 40])

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

    data_np, node_order = load_npz_with_ids(npz_path)
    num_nodes = data_np.shape[1]
    in_channels = data_np.shape[2]
    print(f"[INFO] data shape = {data_np.shape}, nodes={num_nodes}, channels={in_channels}")

    train_np, _, _ = split_data(data_np)
    x_tr_np, y_tr_np = generate_windows(train_np, args.t_in, args.t_out)

    scaler = StandardScaler()
    scaler.fit(torch.from_numpy(train_np))
    x_tr = scaler.transform(torch.from_numpy(x_tr_np))
    y_tr = scaler.transform(torch.from_numpy(y_tr_np))

    A = build_adj_from_csv(csv_path, name_upper, node_order=node_order)
    A_tensor = torch.tensor(A, dtype=torch.float32, device=args.device)

    # split to clients
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

    # model factory
    def build_model():
        m = DSGFL_Wrapper(
            num_nodes=num_nodes,
            in_channels=in_channels,
            d_model=96,
            horizon=args.t_out,
            cheb_K=args.cheb_K,
            mem_slots=16,
            diff_L=3,
            dropout=0.10,
            sem_top_l=4,
            sem_window=4,
            sem_gamma=0.6,
        ).to(args.device)
        m.set_default_adj(A_tensor)
        return m

    def loss_on_norm(pred, yb):
        return torch.mean(torch.abs(pred - yb))

    def opt_ctor(params):
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # proximal schedule (close to main1)
    def eta_schedule(step_idx: int) -> float:
        return float(6e-4 * (0.992 ** step_idx))

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
    runner = FederatedRunner(server=aggregator, clients=fl_clients, tol=0.0, max_rounds=1, verbose=False)

    # run round-by-round and record delta(k)
    deltas = []
    prev_states = [copy.deepcopy(c.model.state_dict()) for c in fl_clients]

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

        cur_states = [copy.deepcopy(c.model.state_dict()) for c in fl_clients]
        d = avg_param_delta(prev_states, cur_states)
        deltas.append({"round": r, "avg_param_delta": d})
        print(f"[Round {r:02d}] avg_param_delta = {d:.6f}")

        prev_states = cur_states

    df = pd.DataFrame(deltas)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"sens_comm_rounds_{name_upper}_K{args.cheb_K}.csv")
    df.to_csv(out_csv, index=False)
    print("\n[SAVED]", out_csv)

    # report specified rounds
    report = df[df["round"].isin(args.report_rounds)].copy()
    report = report.sort_values("round")
    print("\n[REPORT] avg_param_delta at specified rounds:")
    print(report.to_string(index=False))

    # optional plot
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(df["round"].values, df["avg_param_delta"].values, marker="o", markersize=3)
        plt.xlabel("Communication round")
        plt.ylabel("Average parameter delta (L2)")
        plt.grid(True, alpha=0.3)
    except Exception as e:
        print("[WARN] plot skipped:", repr(e))


if __name__ == "__main__":
    main()