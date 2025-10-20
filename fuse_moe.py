from torch import nn
import torch
import torch.nn.functional as F


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
    log_probs = F.log_softmax(router_logits, dim=-1)  # [B, E]

    # Compute entropy H = -sum p*log p (batch of samples)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [B]

    # Only penalize samples with non-zero gates (optional)
    # mask = topk_indices.sum(dim=-1) >= 0
    # entropy = entropy * mask.float()

    # Entropy regularization = -weight * entropy (encourage high entropy)
    ent_loss = entropy.mean(dim=-1) - probs.mean(dim=-1) * torch.log(probs.mean(dim=-1))

    if reduction == "mean":
        return ent_loss.mean()
    elif reduction == "sum":
        return ent_loss.sum()
    else:
        return ent_loss  # [B]


def laplace_gating_with_probs(x, expert_embeddings, k):
    """
    Selects top-K experts for each input based on Laplace gating with per-input expert embeddings.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        expert_embeddings (torch.Tensor): Tensor of expert embeddings of shape (batch_size, num_experts, embedding_dim).
        k (int): Number of top experts to select.

    Returns:
        topk_indices (torch.Tensor): Indices of top-K experts for each input, shape (batch_size, k).
        topk_probs (torch.Tensor): Softmax-normalized probabilities for selected experts, shape (batch_size, k).
    """
    # Ensure x has shape (batch_size, 1, input_dim) for broadcasting
    x_expanded = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)

    # Compute squared Euclidean distances between x and each expert embedding
    distances = torch.sqrt(
        torch.sum((x_expanded - expert_embeddings) ** 2, dim=2)
    )  # Shape: (batch_size, num_experts)

    # Select top-K experts
    topk_scores, topk_indices = torch.topk(
        -distances, k, dim=1
    )  # Both shapes: (batch_size, k)

    # # Apply softmax to top-K scores to get probabilities
    # # print(distances, topk_scores)
    topk_scores = torch.exp(topk_scores)
    topk_probs = topk_scores / (
        torch.sum(topk_scores, dim=1, keepdim=True)
    )

    return topk_indices, topk_probs


class Router(nn.Module):
    def __init__(self, num_experts, hidden_dim):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        self.router = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.repeat_interleave(x, repeats=self.num_experts, dim=1)
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
            nn.Linear(hidden_dim, total_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(total_dim),
        )

    def forward(self, x):
        return self.expert(x)


class MoE(nn.Module):
    def __init__(self, total_dim, hidden_dim, out_dim=512, num_experts=4, k=2, num_modalities=2):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.out_dim = out_dim
        self.num_modalities = num_modalities

        self.router = Router(total_dim, total_dim)

        self.experts = nn.ModuleList(
            [Expert(total_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.out_linear = torch.nn.Sequential(
            torch.nn.Linear(total_dim, out_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        router_embeddings = self.router(x)
        topk_indices, topk_scores = laplace_gating_with_probs(x, router_embeddings, self.k)

        expert_embeddings = []
        for batch_idx, expert_ids in enumerate(topk_indices):
            _embedding = None
            for expert_idx in expert_ids:
                out = self.experts[expert_idx](x[batch_idx]) * topk_scores[batch_idx][expert_idx]
                if _embedding is None:
                    _embedding = out
                else:
                    _embedding = _embedding + out

            expert_embeddings.append(_embedding)

        expert_embeddings = torch.stack(expert_embeddings, dim=0)
        expert_embeddings = expert_embeddings.unsqueeze(1)

        outs = torch.sum(expert_embeddings, dim=1)
        outs = self.out_linear(outs)
        ent_loss = moe_entropy_loss(topk_scores, reduction="mean")

        return outs, ent_loss
