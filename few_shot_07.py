# =========================== few_shot_07.py ===========================
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from main1 import (
    load_npz_with_ids,
    build_adj_from_csv,
    split_data,
    generate_windows,
    StandardScaler,
    DSGFL_Wrapper,
    masked_mae,
    masked_rmse,
    masked_mape,
)
from FL import FLClient, FLAggregator, FederatedRunner

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


@torch.no_grad()
def evaluate(global_model, scaler, x_te_np, y_te_np, device="cuda", batch_size=64):
    global_model.eval()
    x_te = scaler.transform(torch.from_numpy(x_te_np))
    y_te = scaler.transform(torch.from_numpy(y_te_np))

    loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)

    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            out = global_model(xb)  # ✅ wrapper 只收 X
        out_denorm = scaler.inverse_transform(out.detach().cpu())
        yb_denorm = scaler.inverse_transform(yb.detach().cpu())
        preds.append(out_denorm)
        trues.append(yb_denorm)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    mae = masked_mae(pred, true).item()
    rmse = masked_rmse(pred, true).item()
    mape = masked_mape(pred, true).item()
    return mae, rmse, mape


def run_few_shot_experiment(
    root,
    dataset="PEMS07",
    device="cuda",
    t_in=12,
    t_out=12,
    ratios=(0.1, 0.2, 1.0),
    rounds=30,
    clients=3,
    batch=16,
    lr=4e-4,
    weight_decay=1e-3,
    max_batches_per_client=60,
    tol=1e-4,
    seed=42,
):
    dataset = str(dataset).upper()
    folder = os.path.join(root, dataset)
    npz_path = os.path.join(folder, f"{dataset}.npz")
    csv_path = os.path.join(folder, f"{dataset}.csv")

    data_np, node_order = load_npz_with_ids(npz_path)
    print(f"[INFO] raw data shape = {data_np.shape}")

    num_nodes = data_np.shape[1]
    in_channels = data_np.shape[2]

    train_np, _, test_np = split_data(data_np)
    x_tr_np, y_tr_np = generate_windows(train_np, t_in, t_out)
    x_te_np, y_te_np = generate_windows(test_np, t_in, t_out)
    print(f"[INFO] full-train windows={x_tr_np.shape}, test windows={x_te_np.shape}")

    A = build_adj_from_csv(csv_path, dataset, node_order)
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    scaler = StandardScaler()
    scaler.fit(torch.from_numpy(train_np))

    def build_model():
        m = DSGFL_Wrapper(num_nodes=num_nodes, in_channels=in_channels, horizon=t_out).to(device)
        m.set_default_adj(A_tensor)
        return m

    def loss_on_norm(pred, yb):
        return torch.mean(torch.abs(pred - yb))

    def opt_ctor(params):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def eta_schedule(step_idx: int) -> float:
        return float(1.2e-3 * (0.985 ** step_idx))

    results = {}

    for ratio in ratios:
        ratio = float(ratio)
        n_all = x_tr_np.shape[0]
        n_use = max(32, int(n_all * ratio))

        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_all)[:n_use]
        idx = np.sort(idx)

        x_tr_sel = torch.from_numpy(x_tr_np[idx])
        y_tr_sel = torch.from_numpy(y_tr_np[idx])

        x_tr = scaler.transform(x_tr_sel)
        y_tr = scaler.transform(y_tr_sel)

        # split to clients
        idx_all = np.arange(n_use)
        client_indices = np.array_split(idx_all, clients)

        def make_loader(x, y):
            ds = TensorDataset(x.to(torch.float32), y.to(torch.float32))
            return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)

        train_loaders = [make_loader(x_tr[ci], y_tr[ci]) for ci in client_indices]

        client_models = [build_model() for _ in range(clients)]
        global_model = build_model()

        fl_clients = [
            FLClient(
                client_id=i,
                model=client_models[i],
                loss_fn=loss_on_norm,
                optimizer_ctor=opt_ctor,
                device=torch.device(device),
                use_amp=True,
                grad_clip=3.0,
                verbose=False,
            )
            for i in range(clients)
        ]

        aggregator = FLAggregator(sigma=8.0, tau=1000.0, device=torch.device(device))
        runner = FederatedRunner(server=aggregator, clients=fl_clients, tol=tol, max_rounds=rounds, verbose=True)

        print(f"\n=== Few-shot {dataset} ratio={int(ratio*100)}% (use {n_use} windows) ===")
        runner.run(
            train_loaders=train_loaders,
            local_epochs=1,
            local_lr=lr,
            weight_decay=weight_decay,
            max_batches_per_client=max_batches_per_client,
            stepwise_eta_prox=eta_schedule,
            ema_beta_for_sum=0.1,
        )

        global_model.load_state_dict(fl_clients[0].model.state_dict(), strict=False)
        mae, rmse, mape = evaluate(global_model, scaler, x_te_np, y_te_np, device=device)
        print(f"[Result] {dataset} {int(ratio*100)}%: MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")
        results[ratio] = (mae, rmse, mape)

    print(f"\n=== Few-shot Summary on {dataset} ===")
    for ratio in ratios:
        mae, rmse, mape = results[float(ratio)]
        print(f"{int(float(ratio)*100)}% data -> MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root dir containing PEMS07/PEMS07.npz and PEMS07.csv")
    parser.add_argument("--dataset", type=str, default="PEMS07")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 1.0])
    args = parser.parse_args()

    run_few_shot_experiment(root=args.root, dataset=args.dataset, device=args.device, ratios=args.ratios)