# =========================== sensitivity_FL.py ===========================
"""
对式 (20) 中的 sigma 和 tau 做灵敏度分析的脚本。

说明：
- 不修改 main1.py / FL.py / DSGL.py / embedding.py
- 通过临时修改 main1.FLAggregator 实现不同 (sigma, tau) 的测试
- 每一组 (sigma, tau) 都完整跑一遍 run_dataset，并记录 MAE / RMSE / MAPE
"""

import os
import argparse
import pandas as pd
import torch

import main1 as m  # 直接导入你的主实验脚本 main1.py

GLOBAL_SIGMA = None
GLOBAL_TAU = None

def run_with_sigma_tau(
    dataset: str,
    root: str,
    device: str = "cuda",
    t_in: int = 12,
    t_out: int = 12,
    sigma: float = 8.0,
    tau: float = 1000.0,
):
    """
    在不改 main1.py 的前提下，用指定的 (sigma, tau) 跑一遍 run_dataset。

    实现方法：
    - 先保存原来的 m.FLAggregator 类
    - 定义一个 PatchedAggregator，继承原类，但在 __init__ 中忽略传入的
      sigma/tau 参数，改用我们想要的值
    - 把 m.FLAggregator 指向 PatchedAggregator
    - 调用 m.run_dataset(...)
    - 最后无论成功/失败，都把 m.FLAggregator 还原
    """
    global GLOBAL_SIGMA, GLOBAL_TAU
    GLOBAL_SIGMA = sigma
    GLOBAL_TAU = tau
    # 1. 备份原始 Aggregator
    OrigAgg = m.FLAggregator

    # 2. 定义一个“打补丁”的 Aggregator
    class PatchedAggregator(OrigAgg):
        def __init__(self, sigma: float, tau: float, device: torch.device, **kwargs):
            # 直接无视 main1.py 传进来的 sigma/tau
            # 强制使用 sensitivity 实验中指定的值
            super().__init__(
                sigma=float(GLOBAL_SIGMA),
                tau=float(GLOBAL_TAU),
                device=device
            )

    # 3. 替换 main1 中的 FLAggregator
    m.FLAggregator = PatchedAggregator

    try:
        # 调用原来的 run_dataset，其他超参数完全不动
        mae, rmse, mape = m.run_dataset(
            name=dataset,
            root=root,
            device=device,
            t_in=t_in,
            t_out=t_out,
        )
    finally:
        # 4. 无论如何都要还原，避免影响后续其他代码
        m.FLAggregator = OrigAgg

    return mae, rmse, mape


def sensitivity_analysis(
    dataset: str,
    root: str,
    device: str = "cuda",
    t_in: int = 12,
    t_out: int = 12,
):
    """
    对式 (20) 的两个超参数 sigma / tau 做一维灵敏度分析：

    1) 固定 tau = 1000.0，扫描 sigma ∈ {2, 4, 6, 8, 10, 12}
    2) 固定 sigma = 8.0，扫描 tau ∈ {500, 800, 1000, 1500, 2000}

    结果：
    - 在 root 下创建 {DATASET}_SENS/ 目录
    - 输出两个 CSV：
        {DATASET}_sigma_sensitivity.csv
        {DATASET}_tau_sensitivity.csv
    """
    name_upper = str(dataset).upper()
    print(f"\n=== Sensitivity analysis on dataset {name_upper} ===")

    out_dir = os.path.join(root, f"{name_upper}_SENS")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) sigma 灵敏度（固定 tau） ----------
    fixed_tau = 1000.0
    sigma_list = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    sigma_records = []

    print(f"\n>>> Sweep sigma with fixed tau = {fixed_tau:.1f}")
    for sig in sigma_list:
        print(f"\n[Config] sigma = {sig:.1f}, tau = {fixed_tau:.1f}")
        mae, rmse, mape = run_with_sigma_tau(
            dataset=dataset,
            root=root,
            device=device,
            t_in=t_in,
            t_out=t_out,
            sigma=sig,
            tau=fixed_tau,
        )
        print(f"Result: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

        sigma_records.append(
            {
                "sigma": sig,
                "tau": fixed_tau,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
            }
        )

    df_sigma = pd.DataFrame(sigma_records)
    out_sigma = os.path.join(out_dir, f"{name_upper}_sigma_sensitivity.csv")
    df_sigma.to_csv(out_sigma, index=False)
    print(f"\n[Saved] Sigma sensitivity results -> {out_sigma}")

    # ---------- 2) tau 灵敏度（固定 sigma） ----------
    fixed_sigma = 8.0
    tau_list = [500.0, 800.0, 1000.0, 1500.0, 2000.0]
    tau_records = []

    print(f"\n>>> Sweep tau with fixed sigma = {fixed_sigma:.1f}")
    for tau in tau_list:
        print(f"\n[Config] sigma = {fixed_sigma:.1f}, tau = {tau:.1f}")
        mae, rmse, mape = run_with_sigma_tau(
            dataset=dataset,
            root=root,
            device=device,
            t_in=t_in,
            t_out=t_out,
            sigma=fixed_sigma,
            tau=tau,
        )
        print(f"Result: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

        tau_records.append(
            {
                "sigma": fixed_sigma,
                "tau": tau,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
            }
        )

    df_tau = pd.DataFrame(tau_records)
    out_tau = os.path.join(out_dir, f"{name_upper}_tau_sensitivity.csv")
    df_tau.to_csv(out_tau, index=False)
    print(f"\n[Saved] Tau sensitivity results -> {out_tau}")

    print("\n=== Sensitivity analysis finished ===")


# -------------------- main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="数据所在根目录（与 main1.py 参数一致）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称，例如：PEMS03 / PEMS04 / PEMS07 / PEMS08 / HN 等",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备：cuda 或 cpu，默认 cuda",
    )
    parser.add_argument(
        "--t_in",
        type=int,
        default=12,
        help="历史时间步长度（与主实验保持一致）",
    )
    parser.add_argument(
        "--t_out",
        type=int,
        default=12,
        help="预测时间步长度（与主实验保持一致）",
    )

    args = parser.parse_args()

    sensitivity_analysis(
        dataset=args.dataset,
        root=args.root,
        device=args.device,
        t_in=args.t_in,
        t_out=args.t_out,
    )
