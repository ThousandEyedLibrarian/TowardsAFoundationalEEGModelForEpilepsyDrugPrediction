import torch
import torch.nn as nn
import torch.nn.functional as F

class RunningMarginalMI(nn.Module):
    def __init__(self, num_groups: int, num_codes: int = None, momentum: float = 0.99, eps: float = 1e-9, device='cpu'):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.num_groups = num_groups
        self.num_codes = num_codes
        if num_codes is not None:
            self.register_buffer(
                "running_marginal",
                torch.ones(num_groups, num_codes, device=device) / num_codes
            )
        else:
            self.register_buffer(
                "running_marginal",
                torch.ones(num_groups, device=device) / num_groups
            )

    def normalize_marginal(self, m, eps=1e-8):
        m = m.clamp(min=eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        return m

    @torch.no_grad()
    def update_running_marginal(self, p: torch.Tensor):
        """
        Updates running marginal using current batch.
        Args:
            p: [B, T, G, V] or [B, T, K]
        """
        if p.dim() == 4:
            B, T, G, V = p.shape
            if self.running_marginal.numel() == 1:
                # Initialize properly on first call
                self.num_groups, self.num_codes = G, V
                init_value = torch.ones(G, V, device=p.device) / V
                self.running_marginal = init_value
            batch_mean = p.mean(dim=(0, 1))  # [G, V]

        elif p.dim() == 3:
            B, T, K = p.shape
            if self.running_marginal.numel() == 1:
                # Initialize properly on first call
                self.num_codes = K
                self.running_marginal = torch.ones(K, device=p.device) / K
            batch_mean = p.mean(dim=(0, 1))  # [K]

        else:
            raise ValueError(f"Unsupported input shape {p.shape}")

        # EMA update
        self.running_marginal.mul_(self.momentum).add_((1.0 - self.momentum) * (batch_mean + self.eps))
        self.running_marginal = self.normalize_marginal(self.running_marginal, self.eps)

    def get_marginal(self):
        """
        Returns normalized running marginal.
        """
        self.running_marginal = torch.clamp(self.running_marginal, min=self.eps)
        marginal = self.running_marginal / (self.running_marginal.sum(dim=-1, keepdim=True) + self.eps)
        return marginal

    def compute_mi_loss(self, p: torch.Tensor, return_diagnostics: bool = False):
        """
        Computes MI-style diversity loss: E[H(p)] - H(E[p])
        """
        self.update_running_marginal(p)
        marginal = self.get_marginal()

        # Per-sample entropy
        log_p = torch.log(p + self.eps)
        per_sample_entropy = -torch.sum(p * log_p, dim=-1)
        mean_entropy = per_sample_entropy.mean()

        # Entropy of the marginal
        log_marginal = torch.log(marginal + self.eps)
        entropy_of_mean = -torch.sum(marginal * log_marginal, dim=-1).mean()

        mi_loss = mean_entropy - entropy_of_mean

        result = {'mi_loss': mi_loss}

        if return_diagnostics:
            approx_codes_used = (marginal > 1e-3).sum().item()
            result.update({
                'mean_entropy': mean_entropy.detach().item(),
                'entropy_of_mean': entropy_of_mean.detach().item(),
                'approx_codes_used': approx_codes_used
            })

        return result
