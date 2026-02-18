import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Sequence, Union
from datetime import datetime
import pandas as pd


def temporal_positional_encoding(T: int, d_model: int, device=None) -> torch.FloatTensor:
    pe = torch.zeros(T, d_model, device=device)
    position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class ChebSpatialEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 3, bias: bool = True):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta = nn.Parameter(torch.randn(K, in_channels, out_channels) *
                                  (2.0 / math.sqrt(in_channels + out_channels)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # 缓存（减少重复特征分解开销）
        self.register_buffer("_cached_tildeL", torch.tensor(0.0), persistent=False)
        self._cached_shape = None
        self._cached_dtype = None
        self._cached_device = None
        self._tildeL_ready = False

    @staticmethod
    def normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
        N = A.shape[0]
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(N, device=A.device, dtype=A.dtype) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

    @staticmethod
    def _estimate_lambda_max(L: torch.Tensor, iters: int = 20) -> float:
        N = L.size(0)
        v = torch.randn(N, device=L.device, dtype=L.dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            v = L @ v
            v = v / (v.norm() + 1e-12)
        lam = torch.dot(v, L @ v)
        lam = torch.clamp(lam, min=1e-6, max=2.0)
        return float(lam.item())

    def _maybe_update_tildeL(self, A: torch.Tensor, dtype, device):
        need_update = (
            (not self._tildeL_ready) or
            (self._cached_shape != tuple(A.shape)) or
            (self._cached_dtype != dtype) or
            (self._cached_device != device)
        )
        if need_update:
            L = self.normalized_laplacian(A.to(device=device, dtype=dtype))
            lam_max = self._estimate_lambda_max(L)
            tildeL = (2.0 / lam_max) * L - torch.eye(L.size(0), device=device, dtype=dtype)
            self._cached_tildeL = tildeL
            self._cached_shape = tuple(A.shape)
            self._cached_dtype = dtype
            self._cached_device = device
            self._tildeL_ready = True

    def forward(self, Z: torch.Tensor, A: torch.Tensor):
        device = Z.device
        dtype = Z.dtype
        self._maybe_update_tildeL(A, dtype, device)
        tildeL = self._cached_tildeL

        def cheb_apply(X):
            T_list = []
            T0 = X
            T_list.append(T0)
            if self.K > 1:
                T1 = torch.einsum("ij,tjd->tid", tildeL, X)
                T_list.append(T1)
                for _k in range(2, self.K):
                    Tk = 2.0 * torch.einsum("ij,tjd->tid", tildeL, T_list[-1]) - T_list[-2]
                    T_list.append(Tk)
            out = 0.0
            for k in range(self.K):
                out = out + torch.einsum("tni,io->tno", T_list[k], self.theta[k])
            if self.bias is not None:
                out = out + self.bias
            return out

        if Z.dim() == 2:
            Zt = Z.unsqueeze(0)
            out = cheb_apply(Zt).squeeze(0)
            return out
        elif Z.dim() == 3:
            return cheb_apply(Z)
        else:
            raise ValueError("Z must be 2D or 3D")


class TemporalPeriodicEmbed(nn.Module):
    def __init__(self, d_model: int, M: int = 5, max_minutes=1440):
        super().__init__()
        self.d_model = d_model
        self.M = M
        self.periods_per_day = max_minutes // M
        self.emb_day = nn.Embedding(self.periods_per_day, d_model)
        self.emb_week = nn.Embedding(7, d_model)

    @staticmethod
    def timestamps_to_indices(timestamps: Sequence[Union[datetime, str, pd.Timestamp]], M: int):
        minute_idx, weekday_idx = [], []
        for ts in timestamps:
            if not isinstance(ts, datetime):
                ts = pd.to_datetime(ts).to_pydatetime()
            minute_bucket = int((ts.hour * 60 + ts.minute) // M)
            minute_idx.append(minute_bucket)
            weekday_idx.append(int(ts.weekday()))
        return torch.tensor(minute_idx, dtype=torch.long), torch.tensor(weekday_idx, dtype=torch.long)

    def forward(self, T: int, minute_idx: Optional[torch.LongTensor] = None,
                weekday_idx: Optional[torch.LongTensor] = None,
                timestamps: Optional[Sequence[Union[datetime, str, pd.Timestamp]]] = None,
                device=None):
        if timestamps is not None:
            minute_idx, weekday_idx = self.timestamps_to_indices(timestamps, M=self.M)

        if minute_idx is None or weekday_idx is None:
            E_d = torch.zeros(T, self.d_model, device=device)
            E_w = torch.zeros(T, self.d_model, device=device)
            return E_d, E_w

        minute_idx = minute_idx.to(device)
        weekday_idx = weekday_idx.to(device)
        minute_idx = torch.remainder(minute_idx, self.emb_day.num_embeddings)
        weekday_idx = torch.remainder(weekday_idx, 7)

        E_d = self.emb_day(minute_idx)  # (T,d)
        E_w = self.emb_week(weekday_idx)  # (T,d)
        return E_d, E_w


class TrafficEmbedding(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 64, M: int = 5, cheb_K: int = 3, device=None):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()
        self.temporal_periodic = TemporalPeriodicEmbed(d_model, M)
        self.cheb = ChebSpatialEmbed(in_channels=d_model, out_channels=d_model, K=cheb_K)
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor, A: Union[torch.Tensor, np.ndarray],
                minute_idx: Optional[torch.LongTensor] = None,
                weekday_idx: Optional[torch.LongTensor] = None,
                timestamps: Optional[Sequence[Union[datetime, str, pd.Timestamp]]] = None):
        device = X.device
        B, T, N, _ = X.shape

        # Linear map
        Xp = self.input_proj(X)

        # Periodic temporal embeddings
        E_d, E_w = self.temporal_periodic(T=T, minute_idx=minute_idx, weekday_idx=weekday_idx,
                                          timestamps=timestamps, device=device)  # (T,d)

        def _broadcast_temporal(E: torch.Tensor) -> torch.Tensor:
            return E.view(1, T, 1, self.d_model).expand(B, T, N, self.d_model)

        E_d = _broadcast_temporal(E_d)
        E_w = _broadcast_temporal(E_w)
        E_p = temporal_positional_encoding(T, self.d_model, device=device)
        E_p = _broadcast_temporal(E_p)

        if isinstance(A, np.ndarray):
            A_t = torch.tensor(A, dtype=Xp.dtype, device=device)
        else:
            A_t = A.to(device=device, dtype=Xp.dtype)

        E_s = torch.stack([self.cheb(Xp[b], A_t) for b in range(B)], dim=0)  # (B,T,N,d)

        # Final fused embeddings
        X_te = self.norm(self.dropout(Xp + E_d + E_w + E_p))
        X_sp = self.norm(self.dropout(Xp + E_s + E_p))

        return X_te, X_sp
