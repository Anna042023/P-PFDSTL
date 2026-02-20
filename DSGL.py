
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def normalize_symmetric_adj(A: torch.Tensor) -> torch.Tensor:
    A = A.float()
    deg = A.sum(-1) + 1e-12
    D_inv_sqrt = torch.pow(deg, -0.5)
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)


class GlobalTemporalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, N, d = X.shape
        X_bn = X.permute(0, 2, 1, 3).contiguous().view(B * N, T, d)

        Q = self.W_Q(X_bn)
        K = self.W_K(X_bn)
        V = self.W_V(X_bn)

        def split(x):
            return x.view(B * N, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        Qh, Kh, Vh = split(Q), split(K), split(V)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)
        Fm = torch.softmax(scores, dim=-1)
        Fm = self.dropout(Fm)
        out_h = torch.matmul(Fm, Vh)
        out = out_h.permute(0, 2, 1, 3).contiguous().view(B * N, T, d)
        out = self.out(out)
        out = out + X_bn
        out = self.norm(out)

        out = out.view(B, N, T, d).permute(0, 2, 1, 3).contiguous()
        return out


class LocalTemporalCausalConv(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                              kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, N, d = X.shape
        x_bn = X.permute(0, 2, 3, 1).contiguous().view(B * N, d, T)
        x_bn = F.pad(x_bn, (self.left_padding, 0))
        y = self.conv(x_bn)
        y = self.norm(y)
        y = F.relu(y)
        y = y.view(B, N, d, T).permute(0, 3, 1, 2).contiguous()
        return y


class AdaptiveTemporalFusion(nn.Module):
    def __init__(self, d_model: int, negative_slope: float = 0.01):
        super().__init__()
        self.W_conv = nn.Linear(d_model, d_model, bias=False)
        self.W_att = nn.Linear(d_model, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))
        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H_att: torch.Tensor, H_conv: torch.Tensor) -> torch.Tensor:
        G = torch.sigmoid(self.W_conv(H_conv) + self.W_att(H_att) + self.b)
        H_gate = G * H_conv + (1.0 - G) * H_att
        return self.norm(self.act(H_gate))


class MultiGranularityTemporalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.att = GlobalTemporalAttention(d_model, n_heads=n_heads)
        self.conv = LocalTemporalCausalConv(d_model, kernel_size=kernel_size, dilation=dilation)
        self.fuse = AdaptiveTemporalFusion(d_model)

    def forward(self, X_te: torch.Tensor) -> torch.Tensor:
        H_att = self.att(X_te)
        H_conv = self.conv(X_te)
        H_gate = self.fuse(H_att, H_conv)
        return H_gate


class MultiGranularityTemporalNet(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4,
                 k1: int = 3, dila1: int = 1,
                 k2: int = 3, dila2: int = 2):
        super().__init__()
        self.block1 = MultiGranularityTemporalBlock(d_model, n_heads, k1, dila1)
        self.block2 = MultiGranularityTemporalBlock(d_model, n_heads, k2, dila2)

    def forward(self, X_te: torch.Tensor) -> torch.Tensor:
        H1 = self.block1(X_te)
        H2 = self.block2(H1)
        return H2  # H_temp


def positive_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class SpatialMemoryAttention(nn.Module):
    def __init__(self, d_model: int, E: int = 64, num_memory: int = 16,
                 n_heads: int = 4, node_embed_dim: int = 16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.E = E
        self.M = num_memory
        self.h = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, E, bias=False)
        self.W_K = nn.Linear(E, E, bias=False)
        self.W_V = nn.Linear(E, d_model, bias=False)
        self.W_PSI = nn.Linear(E, d_model, bias=False)

        self.PHI_m = nn.Parameter(torch.randn(self.M, self.E) * (1.0 / math.sqrt(self.E)))
        self.PSI_m = nn.Parameter(torch.randn(self.M, self.E) * (1.0 / math.sqrt(self.E)))

        self.node_embed_dim = node_embed_dim   # 保留但不再用于偏置图
        self.leaky = nn.LeakyReLU(0.01)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)

        # 仍然注册以保持兼容，但后续不使用 A_bias
        self.node_E1 = None
        self.node_E2 = None

    def _ensure_node_embeds(self, N: int, device, dtype):
        if (self.node_E1 is None) or (self.node_E1.shape[0] != N):
            self.node_E1 = nn.Parameter(torch.randn(N, self.node_embed_dim, device=device, dtype=dtype) * 0.02)
            self.node_E2 = nn.Parameter(torch.randn(N, self.node_embed_dim, device=device, dtype=dtype) * 0.02)
            self.register_parameter("node_E1", self.node_E1)
            self.register_parameter("node_E2", self.node_E2)

    def forward(self, H_temp: torch.Tensor, A_static: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, d = H_temp.shape
        device, dtype = H_temp.device, H_temp.dtype
        self._ensure_node_embeds(N, device, dtype)

        # 记忆槽键/值/调制
        K = self.W_K(self.PHI_m)                  # (M, E)
        V = self.W_V(self.PHI_m)                  # (M, d)
        Psi = self.W_PSI(self.PSI_m)              # (M, d)

        # 线性注意核化
        phi_K = positive_feature_map(K)           # (M, E)
        KtV = torch.einsum("me,md->ed", phi_K, V) # (E, d)

        # 查询
        H_bt = H_temp.contiguous().view(B * T, N, d)  # (BT, N, d)
        Q = self.W_Q(H_bt)                             # (BT, N, E)
        phi_Q = positive_feature_map(Q)                # (BT, N, E)

        # 记忆注意权重
        QK = torch.einsum("bne,me->bnm", phi_Q, phi_K) / math.sqrt(self.E)  # (BT, N, M)
        A_dyn_qm = torch.softmax(QK, dim=-1)                                 # (BT, N, M)

        # 主通路注意输出
        main_att = torch.einsum("bne,ed->bnd", phi_Q, KtV)                   # (BT, N, d)
        mem_val = torch.einsum("bnm,md->bnd", A_dyn_qm, V)                   # (BT, N, d)


        att_out = main_att  # + 0


        H_prime = att_out

        # 残差 + 投影 + 归一化
        H_prime = H_prime + H_bt
        H_prime = self.out_proj(H_prime)
        H_prime = self.norm(H_prime).view(B, T, N, d)

        # 动态图（按原实现保持）
        mem_val_bt = mem_val.view(B, T, N, d).mean(dim=1)                    # (B, N, d)
        A_dynamic = torch.einsum("bnd,bmd->bnm", mem_val_bt, mem_val_bt) / math.sqrt(d)
        A_dynamic = torch.softmax(A_dynamic, dim=-1)

        return H_prime, A_dynamic


class DiffusionSpatialAggregator(nn.Module):
    def __init__(self, d_model: int, L: int = 2, dropout: float = 0.1):
        super().__init__()
        self.L = L
        self.proj = nn.Conv1d(in_channels=d_model * (L + 1), out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H_prime: torch.Tensor, A_fused: torch.Tensor) -> torch.Tensor:
        B, T, N, d = H_prime.shape
        X0 = H_prime.contiguous().view(B * T, N, d)
        diffs = [X0]
        Xk = X0
        A_rep = A_fused.unsqueeze(1).repeat(1, T, 1, 1).contiguous().view(B * T, N, N)
        for _ in range(1, self.L + 1):
            Xk = torch.bmm(A_rep, Xk)
            diffs.append(Xk)
        cat = torch.cat(diffs, dim=-1)
        cat = cat.permute(0, 2, 1).contiguous()
        out = self.proj(cat)
        out = self.dropout(out)
        out = out.permute(0, 2, 1).contiguous().view(B, T, N, d)
        return self.norm(out)


class DynamicMemorySpatialGraph(nn.Module):
    def __init__(self, d_model: int, E: int = 64, num_memory: int = 16,
                 n_heads: int = 4, node_embed_dim: int = 16, L: int = 2, dropout: float = 0.1):
        super().__init__()
        self.sma = SpatialMemoryAttention(d_model, E, num_memory, n_heads, node_embed_dim)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.agg = DiffusionSpatialAggregator(d_model, L=L, dropout=dropout)

    def forward(self, H_temp: torch.Tensor, A_static: Optional[torch.Tensor] = None):
        B, T, N, d = H_temp.shape
        H_prime, A_dynamic = self.sma(H_temp, A_static)

        if A_static is None:
            A_fused = A_dynamic
        else:
            A_hat = normalize_symmetric_adj(A_static.to(H_temp.device))
            A_hat_b = A_hat.unsqueeze(0).expand(B, -1, -1).contiguous()
            beta = torch.clamp(self.beta, 0.0, 1.0)
            A_fused = beta * A_dynamic + (1.0 - beta) * A_hat_b

        H_spatial = self.agg(H_prime, A_fused)
        return H_spatial, A_dynamic, A_fused


class SoftDTW(nn.Module):
    def __init__(self, gamma: float = 0.5, window: int = 4):
        super().__init__()
        self.gamma = gamma
        self.window = window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        D = torch.cdist(x, y, p=2)
        W = self.window
        inf = torch.tensor(float('inf'), device=x.device, dtype=x.dtype)
        R = torch.full((T + 1, T + 1), inf, device=x.device, dtype=x.dtype)
        R[0, 0] = 0.0
        for i in range(1, T + 1):
            j_lo = max(1, i - W)
            j_hi = min(T, i + W)
            r0 = R[i - 1, j_lo - 1: j_hi]
            r1 = R[i,     j_lo - 1: j_hi]
            r2 = R[i - 1, j_lo:     j_hi + 1]
            stacked = torch.stack([-r0 / self.gamma, -r1 / self.gamma, -r2 / self.gamma], dim=0)
            softmin = -self.gamma * torch.logsumexp(stacked, dim=0)
            R[i, j_lo: j_hi + 1] = D[i - 1, j_lo - 1: j_hi] + softmin
        return R[T, T]


class SoftDTWSemanticAttention(nn.Module):
    def __init__(self, d_model: int, top_l: int = 5, gamma: float = 0.5, window: int = 4):
        super().__init__()
        self.d_model = d_model
        self.top_l = top_l
        self.soft_dtw = SoftDTW(gamma=gamma, window=window)
        self.window = window
        # 下列线性层不再使用，保留以兼容已有权重加载场景
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    @torch.no_grad()
    def _softdtw_matrix(self, X_sp: torch.Tensor) -> torch.Tensor:
        B, T, N, d = X_sp.shape
        device, dtype = X_sp.device, X_sp.dtype
        C_list = []

        iu = torch.triu_indices(N, N, offset=0, device=device)
        I, J = iu[0], iu[1]
        P = I.numel()

        W = self.window
        inf = torch.tensor(float('inf'), device=device, dtype=dtype)

        for b in range(B):
            X_b = X_sp[b].permute(1, 0, 2).contiguous()  # (N, T, d)
            A = X_b[I]                                   # (P, T, d)
            Bseq = X_b[J]                                # (P, T, d)

            A2 = A.unsqueeze(2)                          # (P, T, 1, d)
            B2 = Bseq.unsqueeze(1)                       # (P, 1, T, d)
            D = torch.norm(A2 - B2, dim=-1)              # (P, T, T)

            R = torch.full((P, T + 1, T + 1), inf, device=device, dtype=dtype)
            R[:, 0, 0] = 0.0
            for i in range(1, T + 1):
                j_lo = max(1, i - W)
                j_hi = min(T, i + W)
                r0 = R[:, i - 1, j_lo - 1: j_hi]
                r1 = R[:, i,     j_lo - 1: j_hi]
                r2 = R[:, i - 1, j_lo:     j_hi + 1]
                stacked = torch.stack([-r0, -r1, -r2], dim=0) / self.soft_dtw.gamma
                softmin = -self.soft_dtw.gamma * torch.logsumexp(stacked, dim=0)
                R[:, i, j_lo: j_hi + 1] = D[:, i - 1, j_lo - 1: j_hi] + softmin

            dist = R[:, T, T]                            # (P,)
            Cb = torch.zeros(N, N, device=device, dtype=dtype)
            Cb[I, J] = dist
            Cb[J, I] = dist
            C_list.append(Cb.unsqueeze(0))

        return torch.cat(C_list, dim=0)                  # (B, N, N)

    @staticmethod
    def _topl_mask(C: torch.Tensor, l: int) -> torch.Tensor:
        B, N, _ = C.shape
        C = C.clone()
        eye = torch.eye(N, device=C.device, dtype=C.dtype).unsqueeze(0)
        C = C + eye * 1e9
        idx = torch.topk(-C, k=l, dim=-1).indices
        M = torch.zeros_like(C)
        b_ids = torch.arange(B, device=C.device)[:, None, None].expand(B, N, l)
        n_ids = torch.arange(N, device=C.device)[None, :, None].expand(B, N, l)
        M[b_ids, n_ids, idx] = 1.0
        return M

    def forward(self, H_temp: torch.Tensor, X_sp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, d = H_temp.shape

        # 1) SoftDTW 距离
        C_soft = self._softdtw_matrix(X_sp)             # (B, N, N), 距离越小越相似

        # 2) 相似度图（去掉 Attention，仅基于 SoftDTW）
        #    行 softmax(-dist) 得到相似度，再用 top-l 稀疏化并重归一化
        S = torch.softmax(-C_soft, dim=-1)              # (B, N, N)
        M_sem = self._topl_mask(C_soft, self.top_l)     # (B, N, N)，选近邻
        A_sem = S * M_sem
        row_sum = A_sem.sum(-1, keepdim=True) + 1e-12
        A_sem = A_sem / row_sum

        H_sem = torch.zeros_like(H_temp)

        return H_sem, A_sem


class FusionAndOutput(nn.Module):
    def __init__(self, d_model: int, z: int = 12, out_dim: int = 1, negative_slope: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.z = z
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(in_channels=3 * d_model, out_channels=d_model, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=z * out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H_temp: torch.Tensor, H_spatial: torch.Tensor, H_sem: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([H_temp, H_spatial, H_sem], dim=-1)
        B, T, N, C3 = fused.shape
        x = fused.permute(0, 2, 3, 1).contiguous().view(B * N, C3, T)
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        y_last = y[..., -1]
        y_last = y_last.view(B, N, self.z, self.out_dim).permute(0, 2, 1, 3).contiguous()
        return y_last


class STFeatureLearner(nn.Module):
    def __init__(self,
                 N: int,
                 d_model: int = 64,
                 H_future: int = 12,
                 out_dim: int = 1,
                 # temporal
                 t_heads: int = 4,
                 t_k1: int = 3, t_dila1: int = 1,
                 t_k2: int = 3, t_dila2: int = 2,
                 # memory spatial
                 mem_E: int = 64, mem_slots: int = 16, mem_heads: int = 4, node_embed_dim: int = 16,
                 diff_L: int = 2, dropout: float = 0.1,
                 # semantic soft-dtw
                 sem_top_l: int = 5, sem_gamma: float = 0.5, sem_window: int = 4):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.temporal_net = MultiGranularityTemporalNet(
            d_model=d_model, n_heads=t_heads,
            k1=t_k1, dila1=t_dila1, k2=t_k2, dila2=t_dila2
        )
        self.spatial_net = DynamicMemorySpatialGraph(
            d_model=d_model, E=mem_E, num_memory=mem_slots,
            n_heads=mem_heads, node_embed_dim=node_embed_dim,
            L=diff_L, dropout=dropout
        )
        self.semantic_net = SoftDTWSemanticAttention(
            d_model=d_model, top_l=sem_top_l, gamma=sem_gamma, window=sem_window
        )
        self.fusion_head = FusionAndOutput(d_model=d_model, z=H_future, out_dim=out_dim)

    def forward(self, X_te: torch.Tensor, X_sp: torch.Tensor, A_static: Optional[torch.Tensor] = None):
        H_temp = self.temporal_net(X_te)
        H_spatial, A_dynamic, A_fused = self.spatial_net(H_temp, A_static=A_static)
        H_sem, A_sem = self.semantic_net(H_temp, X_sp)
        y_pred = self.fusion_head(H_temp, H_spatial, H_sem)
        diagnostics = {
            "H_temp":    H_temp.detach().cpu(),
            "H_spatial": H_spatial.detach().cpu(),
            "H_sem":     H_sem.detach().cpu(),
            "A_dynamic": A_dynamic.detach().cpu(),
            "A_fused":   A_fused.detach().cpu(),
            "A_sem":     A_sem.detach().cpu()
        }
        return y_pred, diagnostics
