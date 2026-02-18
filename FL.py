# =========================== FL.py ===========================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Callable, Optional
import copy, math
import torch
import torch.nn as nn
import torch.optim as optim

StateDict = Dict[str, torch.Tensor]

# -------- utils --------
def _flat_params(state: StateDict) -> torch.Tensor:
    vecs = []
    for v in state.values():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            vecs.append(v.detach().reshape(-1).cpu().float())
    return torch.cat(vecs) if vecs else torch.empty(0)

def _add(a: StateDict, b: StateDict) -> StateDict:
    out = {}
    for k in a.keys():
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor) and va.dtype.is_floating_point and vb.dtype.is_floating_point:
            out[k] = va + vb
        else:
            out[k] = va.clone()
    return out

def _sub(a: StateDict, b: StateDict) -> StateDict:
    out = {}
    for k in a.keys():
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor) and va.dtype.is_floating_point and vb.dtype.is_floating_point:
            out[k] = va - vb
        else:
            out[k] = va.clone()
    return out

def _scale(a: StateDict, w: float) -> StateDict:
    out = {}
    for k, v in a.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            out[k] = v * w
        else:
            out[k] = v.clone()
    return out

def _zeros_like(a: StateDict) -> StateDict:
    out = {}
    for k, v in a.items():
        out[k] = torch.zeros_like(v) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point else v.clone()
    return out

# --------- grab grad dict aligned with state_dict ---------
def _grab_grad_state(model: nn.Module) -> StateDict:
    grads: StateDict = {}
    param_dict = dict(model.named_parameters())
    for name, t in model.state_dict().items():
        if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
            p = param_dict.get(name, None)
            if p is not None and p.grad is not None:
                grads[name] = p.grad.detach().clone()
            else:
                grads[name] = torch.zeros_like(t)
        else:
            grads[name] = t.clone() if isinstance(t, torch.Tensor) else t
    return grads

# ============================= Client =============================
@dataclass
class FLClient:
    """
    Paper-style local update (Eq.19):
        - Online gradient descent on local loss.
        - Proximal refinement w.r.t. cumulative (EMA) gradient S_t.
          theta_new = theta_tilde - eta_prox * S_t
    """
    client_id: int
    model: nn.Module
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizer_ctor: Callable[[Iterable[torch.nn.Parameter]], optim.Optimizer]
    device: torch.device

    cum_grad_state: Optional[StateDict] = None
    eta_prox: float = 1e-3  # safe default (actual per-step由调度器给出)

    # optional DP
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.0
    dp_rng: Optional[torch.Generator] = None

    use_amp: bool = True
    grad_clip: float = 3.0
    verbose: bool = False

    def _prox_exact(self, theta_tilde: StateDict) -> StateDict:
        assert self.cum_grad_state is not None, "cum_grad_state must be initialized."
        out = {}
        for k, v in theta_tilde.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                out[k] = v - self.eta_prox * self.cum_grad_state[k]
            else:
                out[k] = v
        return out

    def _cum_grad_update(self, grad_state: StateDict, beta_ema: Optional[float] = 0.1):
        if self.cum_grad_state is None:
            self.cum_grad_state = _zeros_like(grad_state)
        if beta_ema is None:
            for k, v in grad_state.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    self.cum_grad_state[k] = self.cum_grad_state[k] + v
        else:
            b = float(beta_ema)
            for k, v in grad_state.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    self.cum_grad_state[k] = (1 - b) * self.cum_grad_state[k] + b * v

    def _apply_dp(self, base_state: StateDict, new_state: StateDict) -> StateDict:
        if not self.dp_enabled:
            return new_state
        clip = float(self.dp_clip_norm)
        sigma = float(self.dp_noise_multiplier)
        delta = _sub(new_state, base_state)
        v = _flat_params(delta)
        n = float(torch.norm(v, p=2))
        if n > clip and n > 0:
            delta = _scale(delta, clip / n)
        if sigma > 0:
            for k, t in delta.items():
                if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
                    noise = torch.normal(
                        0.0, sigma * clip, size=t.shape,
                        device=t.device, dtype=t.dtype,
                        generator=self.dp_rng
                    )
                    delta[k] = t + noise
        return _add(base_state, delta)

    def local_update(
        self,
        train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 1,
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        max_batches: Optional[int] = None,
        stepwise_eta_prox: Optional[Callable[[int], float]] = None,
        ema_beta_for_sum: Optional[float] = 0.1,
    ) -> StateDict:
        self.model.to(self.device)
        self.model.train()

        opt = self.optimizer_ctor(self.model.parameters())
        for g in opt.param_groups:
            g["lr"] = lr
            if "weight_decay" in g:
                g["weight_decay"] = weight_decay

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        state_before = copy.deepcopy(self.model.state_dict())

        batch_idx = 0
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)

                scaler.scale(loss).backward()
                if self.grad_clip and self.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(opt)
                scaler.update()

                # 更新累计梯度 S_t
                with torch.no_grad():
                    g_state = _grab_grad_state(self.model)
                    self._cum_grad_update(g_state, beta_ema=ema_beta_for_sum)

                # 近端精炼
                if stepwise_eta_prox is not None:
                    self.eta_prox = float(stepwise_eta_prox(batch_idx))
                with torch.no_grad():
                    theta_tilde = self.model.state_dict()
                    theta_refined = self._prox_exact(theta_tilde)
                    self.model.load_state_dict(theta_refined, strict=False)

                batch_idx += 1
                if max_batches is not None and batch_idx >= max_batches:
                    break
            if max_batches is not None and batch_idx >= max_batches:
                break

        state_after = copy.deepcopy(self.model.state_dict())
        return self._apply_dp(base_state=state_before, new_state=state_after)

# ============================= Aggregator (Eq.20/21) =============================
@dataclass
class FLAggregator:
    """
    Similarity-weighted aggregation:
        w_ij ∝ exp(-||theta_i - theta_j||^2 / (2 sigma^2)) if dist <= tau else 0
    """
    sigma: float = 5.0      # 更平滑但不过分小
    tau: float = 2e3        # 限制过远客户端的影响
    device: torch.device = torch.device("cpu")

    @staticmethod
    def _l2_align(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.flatten().cpu().float()
        b = b.flatten().cpu().float()
        L = min(a.numel(), b.numel())
        if L == 0:
            return float("inf")
        return float(torch.norm(a[:L] - b[:L], p=2).item())

    def _weights(self, states: List[StateDict]) -> torch.Tensor:
        R = len(states)
        flats = [_flat_params(s) for s in states]
        W = torch.zeros((R, R), dtype=torch.float32)
        for i in range(R):
            for j in range(R):
                dist = 0.0 if i == j else self._l2_align(flats[i], flats[j])
                if dist <= self.tau:
                    W[i, j] = math.exp(-(dist ** 2) / (2.0 * (self.sigma ** 2)))
                else:
                    W[i, j] = 0.0
        return W

    def aggregate(self, local_states: List[StateDict]) -> List[StateDict]:
        R = len(local_states)
        W = self._weights(local_states)
        out_states: List[StateDict] = []
        for i in range(R):
            w = W[i]
            s = float(w.sum().item())
            denom = s if s > 0 else 1.0
            acc = None
            for j in range(R):
                sj = local_states[j]
                acc = _scale(sj, float(w[j].item())) if acc is None else _add(acc, _scale(sj, float(w[j].item())))
            out_states.append(_scale(acc, 1.0 / denom))
        return out_states

# ============================= Runner =============================
@dataclass
class FederatedRunner:
    server: FLAggregator
    clients: List[FLClient]
    tol: float = 0.0
    max_rounds: int = 20
    verbose: bool = True

    def _broadcast(self, states: List[StateDict]):
        for c, s in zip(self.clients, states):
            c.model.load_state_dict(s, strict=False)

    def _collect(self) -> List[StateDict]:
        return [copy.deepcopy(c.model.state_dict()) for c in self.clients]

    @torch.no_grad()
    def _avg_param_delta(self, prev: List[StateDict], cur: List[StateDict]) -> float:
        vals = []
        for sp, sc in zip(prev, cur):
            vp, vc = _flat_params(sp), _flat_params(sc)
            L = min(vp.numel(), vc.numel())
            if L > 0:
                vals.append(torch.norm(vp[:L] - vc[:L], p=2).item())
        return float(sum(vals) / max(1, len(vals)))

    def run(
        self,
        train_loaders: List[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
        local_epochs: int = 1,
        local_lr: float = 2e-4,
        weight_decay: float = 1e-3,
        max_batches_per_client: Optional[int] = None,
        stepwise_eta_prox: Optional[Callable[[int], float]] = None,
        ema_beta_for_sum: Optional[float] = 0.1,
    ):
        assert len(train_loaders) == len(self.clients)
        prev_states = self._collect()

        for rnd in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n=== Communication Round {rnd} ===")

            local_states = []
            for i, client in enumerate(self.clients):
                st = client.local_update(
                    train_loader=train_loaders[i],
                    epochs=local_epochs,
                    lr=local_lr,
                    weight_decay=weight_decay,
                    max_batches=max_batches_per_client,
                    stepwise_eta_prox=stepwise_eta_prox,
                    ema_beta_for_sum=ema_beta_for_sum,
                )
                local_states.append(st)

            agg_states = self.server.aggregate(local_states)
            self._broadcast(agg_states)

            cur_states = self._collect()
            delta = self._avg_param_delta(prev_states, cur_states)
            if self.verbose:
                print(f"[Round {rnd}] avg_param_delta = {delta:.6f}")
            if self.tol > 0 and delta < self.tol:
                if self.verbose:
                    print(f"Converged: avg_param_delta < {self.tol}")
                break
            prev_states = cur_states
