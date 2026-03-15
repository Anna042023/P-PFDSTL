# =========================== few_shot_03.py ===========================
import os
import numpy as np
import torch
import random
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


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(global_model, scaler, x_te, y_te, device="cuda", batch_size=64):
    global_model.eval()
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)

    preds, trues = [], []
    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            out = global_model(xb)  # ✅ main1.py 的 wrapper 只收 X
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


def run_few_shot(
    dataset="PEMS03",
    root="./PeMS",
    device="cuda",
    ratios=(0.1, 0.2, 1.0),
    t_in=12,
    t_out=12,
    rounds=25,
    clients=4,
    batch=32,
    lr=5e-4,
    weight_decay=1.5e-3,
    seed=42,
):
    print(f"\n=== 🚦 Few-shot Learning on {dataset} ===")
    set_seed(seed)

    dataset = str(dataset).upper()
    folder = os.path.join(root, dataset)
    npz_path = os.path.join(folder, f"{dataset}.npz")
    csv_path = os.path.join(folder, f"{dataset}.csv")

    # ---------- Load data ----------
    data_np, node_order = load_npz_with_ids(npz_path)
    print(f"[INFO] raw data shape = {data_np.shape}")

    train_np, _, test_np = split_data(data_np)
    x_tr_np, y_tr_np = generate_windows(train_np, t_in=t_in, t_out=t_out)
    x_te_np, y_te_np = generate_windows(test_np, t_in=t_in, t_out=t_out)

    # ---------- Scaler (fit on train only) ----------
    scaler = StandardScaler()
    scaler.fit(torch.from_numpy(train_np))

    x_te = scaler.transform(torch.from_numpy(x_te_np))
    y_te = scaler.transform(torch.from_numpy(y_te_np))

    # ---------- Build adjacency ----------
    A = build_adj_from_csv(csv_path, dataset, node_order)
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    # ---------- Few-shot loop ----------
    results = []
    for ratio in ratios:
        ratio = float(ratio)
        print(f"\n--- Few-shot ratio: {int(ratio * 100)}% ---")

        n_samples = max(32, int(len(x_tr_np) * ratio))
        idx = np.arange(len(x_tr_np))
        np.random.shuffle(idx)
        sel_idx = idx[:n_samples]

        x_tr = scaler.transform(torch.from_numpy(x_tr_np[sel_idx]))
        y_tr = scaler.transform(torch.from_numpy(y_tr_np[sel_idx]))

        # ---------- Federated training ----------
        idx_all = np.arange(len(x_tr))
        client_indices = np.array_split(idx_all, clients)

        def mk_loader(x, y):
            ds = TensorDataset(x.to(torch.float32), y.to(torch.float32))
            return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)

        client_loaders = [mk_loader(x_tr[idx], y_tr[idx]) for idx in client_indices]

        in_channels = data_np.shape[2]
        num_nodes = data_np.shape[1]

        def model_ctor():
            m = DSGFL_Wrapper(
                num_nodes=num_nodes,
                in_channels=in_channels,
                d_model=112,
                horizon=t_out,
                t_heads=4,
                t_k1=3, t_dila1=1,
                t_k2=3, t_dila2=2,
                mem_E=64,
                mem_slots=14,
                mem_heads=4,
                node_embed_dim=16,
                diff_L=3,
                dropout=0.05,
                sem_top_l=4,
                sem_gamma=0.5,
                sem_window=4,
            ).to(device)
            m.set_default_adj(A_tensor)  # ✅ 关键：wrapper 依赖默认邻接
            return m

        client_models = [model_ctor() for _ in range(clients)]
        global_model = model_ctor()

        # ✅ 不再用 to_B_T_N_C1，直接对齐 pred/y
        def loss_fn(pred, yb):
            return torch.mean(torch.abs(pred - yb))

        def opt_ctor(params):
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        fl_clients = [
            FLClient(
                client_id=i,
                model=client_models[i],
                loss_fn=loss_fn,
                optimizer_ctor=opt_ctor,
                device=torch.device(device),
                use_amp=True,
                grad_clip=3.0,
                verbose=False,
            )
            for i in range(clients)
        ]

        aggregator = FLAggregator(sigma=8.0, tau=1000.0, device=torch.device(device))
        runner = FederatedRunner(
            server=aggregator,
            clients=fl_clients,
            tol=0.0,
            max_rounds=rounds,
            verbose=True
        )

        def eta_schedule(step_idx: int) -> float:
            return float(1.2e-3 * (0.985 ** step_idx))

        runner.run(
            train_loaders=client_loaders,
            local_epochs=1,
            local_lr=lr,
            weight_decay=weight_decay,
            stepwise_eta_prox=eta_schedule,
            ema_beta_for_sum=0.1
        )

        # ---------- Evaluation ----------
        global_model.load_state_dict(fl_clients[0].model.state_dict(), strict=False)
        mae, rmse, mape = evaluate(global_model, scaler, x_te, y_te, device=device)
        print(f"[Few-shot {int(ratio*100)}%] MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")
        results.append((ratio, mae, rmse, mape))

    print("\n=== Summary (Few-shot) ===")
    print("Ratio | MAE | RMSE | MAPE")
    print("-----------------------------")
    for r, mae, rmse, mape in results:
        print(f"{int(r*100):>4d}% | {mae:.4f} | {rmse:.4f} | {mape:.2f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root dir containing PEMSxx/PEMSxx.npz and PEMSxx.csv")
    parser.add_argument("--dataset", type=str, default="PEMS03")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 1.0])
    args = parser.parse_args()

    run_few_shot(dataset=args.dataset, root=args.root, device=args.device, ratios=args.ratios)