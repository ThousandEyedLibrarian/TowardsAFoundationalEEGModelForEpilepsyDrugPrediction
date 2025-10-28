from torch import nn
import torch
import torch.nn.functional as F
from running_marginal import RunningMarginalMI
import numpy as np


def moe_entropy_loss(router_logits, reduction="batchmean"):
    """
    Entropy regularization for sparsely gated MoE routers (FuseMoE-style).

    Args:
      router_logits: Tensor [B, E], raw gate logits for each expert e.
      topk_indices: LongTensor [B, K], indices of selected top-K experts per sample.
      reduction: 'batchmean' | 'mean' | 'sum' | 'none'.
      weight: scaling factor β.

    Returns:
      Scalar or [B] entropy loss.
    """
    # Compute full softmax distribution over all experts
    probs = F.softmax(router_logits, dim=-1)  # [B, E]
    log_probs = torch.log(F.softmax(router_logits, dim=-1) + 1e-6)  # [B, E]

    # Compute entropy H = -sum p*log p (batch of samples)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [B]

    # Only penalize samples with non-zero gates (optional)
    # mask = topk_indices.sum(dim=-1) >= 0
    # entropy = entropy * mask.float()

    # Entropy regularization = -weight * entropy (encourage high entropy)
    ent_loss = entropy.mean(dim=-1) - probs.mean(dim=-1) * torch.log(
        probs.mean(dim=-1) + 1e-6
    )

    if reduction == "mean":
        return ent_loss.mean()
    elif reduction == "sum":
        return ent_loss.sum()
    else:
        return ent_loss  # [B]


def laplace_gating_with_probs_timewise(x, router_embedding, k, temperature=0.5):
    """
    Laplace gating per timestep with per-token router embeddings.

    Args:
        x: [B, T, D]
        router_embedding: [B, T, E, D] - one embedding per token per expert
        k: int - top-k experts

    Returns:
        topk_indices: [B, T, k]
        topk_probs: [B, T, k]
    """
    x_exp = x.unsqueeze(2)  # [B, T, 1, D]
    distances = torch.linalg.vector_norm(
        x_exp - router_embedding, dim=-1, ord=2
    )  # [B, T, E]
    distances = torch.exp(-distances / temperature)

    topk_scores, topk_indices = torch.topk(-distances, k, dim=-1)  # [B, T, k]
    topk_probs = topk_scores / torch.sum(topk_scores, dim=-1, keepdim=True)  # [B, T, k]

    return topk_indices, topk_probs


class Router(nn.Module):
    def __init__(self, num_experts, hidden_dim):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.map_num_experts = nn.Linear(hidden_dim, hidden_dim * num_experts)
        self.router = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.map_num_experts(x)
        x = torch.reshape(
            x, (x.shape[0], x.shape[1], self.num_experts, self.hidden_dim)
        )
        x = self.router(x)
        return x


class Expert(nn.Module):
    def __init__(self, total_dim, hidden_dim):
        super(Expert, self).__init__()

        self.total_dim = total_dim
        self.hidden_dim = hidden_dim
        self.expert = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.expert(x)


class MoE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim=512,
        num_experts=4,
        k=2,
        max_temperature=1.0,
        min_temperature=0.5,
        temperature_decay=0.9995,
    ):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.out_dim = out_dim
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.current_temperature = max_temperature
        self.temperature_decay = temperature_decay

        self.router = Router(num_experts, input_dim)
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim) for _ in range(num_experts)]
        )
        print(self.experts[0])
        self.out_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(out_dim),
        )

        self.experts_running_marginal = RunningMarginalMI(
            num_groups=self.num_experts, num_codes=self.k, eps=1e-6
        )

    def apply_topk_experts(self, x, experts, topk_indices, topk_probs):
        """
        Apply top-k experts to each timestep token and combine outputs.

        Args:
            x: [B, T, D] input features
            experts: list or nn.ModuleList of expert networks, each [N, D] -> [N, M]
            topk_indices: [B, T, k] indices of selected experts
            topk_probs: [B, T, k] softmax-normalised gating weights

        Returns:
            combined_output: [B, T, M]
        """
        B, T, D = x.shape
        k = topk_indices.size(-1)
        M = self.hidden_dim  # assume all experts output same dimension

        # Flatten batch and time for token-level processing
        x_flat = x.view(B * T, D)  # [B*T, D]
        topk_idx_flat = topk_indices.view(B * T * k)  # [B*T*k]
        topk_prob_flat = topk_probs.view(B * T * k)  # [B*T*k]

        # Repeat input for each top-k assignment
        x_repeat = (
            x_flat.unsqueeze(1).expand(-1, k, -1).contiguous().view(B * T * k, D)
        )  # [B*T*k, D]

        # Precompute token ids (original token for each top-k assignment)
        token_ids = (
            torch.arange(B * T, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
        )  # [B*T*k]

        # Output buffer
        combined_flat = torch.zeros(B * T, M, device=x.device, dtype=x.dtype)

        # Dispatch per expert
        for e, expert in enumerate(experts):
            mask = topk_idx_flat == e  # [B*T*k]
            if not mask.any():
                continue

            x_e = x_repeat[mask]  # [N_assign, D]
            w_e = topk_prob_flat[mask].unsqueeze(-1)  # [N_assign, 1]
            y_e = expert(x_e)  # [N_assign, M]
            weighted_y = y_e * w_e  # [N_assign, M]

            # Scatter-add to original tokens
            combined_flat.index_add_(0, token_ids[mask], weighted_y)

        # Reshape back to [B, T, M]
        combined = combined_flat.view(B, T, M)
        return combined

    def update_temperature(self, global_step):
        self.current_temperature = np.max(
            [
                self.min_temperature,
                self.max_temperature * np.pow(self.temperature_decay, global_step),
            ]
        )

    def forward(self, x):
        router_embeddings = self.router(x)
        topk_indices, topk_scores = laplace_gating_with_probs_timewise(
            x, router_embeddings, self.k, self.current_temperature
        )

        expert_embeddings = self.apply_topk_experts(
            x, self.experts, topk_indices, topk_scores
        )

        outs = self.out_linear(expert_embeddings)
        ent_loss = self.experts_running_marginal.compute_mi_loss(topk_scores)["mi_loss"]

        return outs, ent_loss
