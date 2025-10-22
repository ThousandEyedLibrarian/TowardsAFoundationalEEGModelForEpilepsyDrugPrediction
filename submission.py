"""
Submission file for EEG Challenge 2025
Optimized S4D Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from typing import Optional, Tuple


def resolve_path(name: str) -> Path:
    """
    Helper function to resolve model weight paths for final phase.

    Checks multiple directories in order:
    1. /app/input/res/{name} (competition evaluation environment)
    2. /app/input/{name} (competition evaluation environment)
    3. Current directory (local testing)
    4. Script's parent directory (local testing)

    Args:
        name: Filename of the weights (e.g., 'weights_challenge_1.pt')

    Returns:
        Path to the weights file

    Raises:
        FileNotFoundError: If weights file not found in any location
    """
    # Possible directories to check (in order of priority)
    dirs = [
        Path("/app/input/res"),      # Competition environment (preferred)
        Path("/app/input"),           # Competition environment (fallback)
        Path.cwd(),                   # Current directory (local testing)
        Path(__file__).parent,        # Script's parent directory (local testing)
    ]

    for d in dirs:
        path = d / name
        if path.exists():
            return path

    # If not found, raise informative error
    raise FileNotFoundError(
        f"Could not find {name} in any of the following locations:\n" +
        "\n".join(f"  - {d}" for d in dirs)
    )


# Import Mixture of Experts module
try:
    from fuse_moe import MoE
    MOE_AVAILABLE = True
except ImportError:
    print("[WARNING] fuse_moe module not found, running without MoE")
    MOE_AVAILABLE = False


# ====================================
# S4D Model Components
# ====================================

class OptimizedS4DKernel(nn.Module):
    """Optimized S4D Kernel with better initialization"""

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Better initialization
        log_dt = torch.linspace(math.log(0.001), math.log(0.1), d_model)
        log_A_real = torch.log(torch.linspace(0.1, 0.9, d_state // 2))
        log_A_real = log_A_real.repeat(d_model, 1)
        A_imag = torch.linspace(0, math.pi, d_state // 2)
        A_imag = A_imag.repeat(d_model, 1)

        C_real = torch.randn(d_model, d_state // 2) / math.sqrt(d_state)
        C_imag = torch.randn(d_model, d_state // 2) / math.sqrt(d_state)

        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)
        self.C = nn.Parameter(torch.stack([C_real, C_imag], dim=-1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt).float()
        C = torch.view_as_complex(self.C.float())
        A = -torch.exp(self.log_A_real.float()) + 1j * self.A_imag.float()

        dtA = A * dt.unsqueeze(1)
        C = C * (torch.exp(dtA) - 1.) / (A + 1e-8)

        vandermonde = torch.exp(dtA.unsqueeze(-1) * torch.arange(L, device=A.device))
        K = 2 * torch.einsum('dn, dnl -> dl', C, vandermonde).real

        return self.dropout(K)


class OptimizedS4DBlock(nn.Module):
    """S4D Block matching architecture diagram"""

    def __init__(self, d_model: int = 128, d_state: int = 64,
                 bidirectional: bool = True, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        self.s4d_kernel = OptimizedS4DKernel(d_model, d_state, dropout)
        if bidirectional:
            self.s4d_kernel_bw = OptimizedS4DKernel(d_model, d_state, dropout)

        self.D = nn.Parameter(torch.randn(d_model) * 0.01)
        self.activation = nn.GELU()

        out_dim = d_model * 2 if bidirectional else d_model
        self.conv1d_out_dim = out_dim if bidirectional else d_model
        self.conv1d = nn.Conv1d(out_dim, self.conv1d_out_dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.norm = nn.LayerNorm(self.conv1d_out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        u = x.transpose(1, 2)

        # Forward kernel
        k = self.s4d_kernel(L)
        k_f = torch.fft.rfft(k, n=2*L)
        u_f = torch.fft.rfft(u, n=2*L)
        y_fw = torch.fft.irfft(u_f * k_f, n=2*L)[..., :L]
        y_fw = y_fw + u * self.D.unsqueeze(-1)

        if self.bidirectional:
            # Backward kernel
            u_bw = torch.flip(u, dims=[-1])
            k_bw = self.s4d_kernel_bw(L)
            k_bw_f = torch.fft.rfft(k_bw, n=2*L)
            u_bw_f = torch.fft.rfft(u_bw, n=2*L)
            y_bw = torch.fft.irfft(u_bw_f * k_bw_f, n=2*L)[..., :L]
            y_bw = y_bw + u_bw * self.D.unsqueeze(-1)
            y_bw = torch.flip(y_bw, dims=[-1])
            y = torch.cat([y_fw, y_bw], dim=1)
        else:
            y = y_fw

        y = self.activation(y)
        y = self.conv1d(y)
        y = self.glu(y)
        y = y.transpose(1, 2)

        if y.shape[-1] == x.shape[-1]:
            y = y + x

        y = self.dropout(y)
        y = self.norm(y)
        return y


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head Attention Pooling"""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x[:, :1, :], x, x)
        return attn_out


class OptimizedS4DEEG(nn.Module):
    """
    Optimized S4D Model for EEG Processing with Optional Demographic Integration

    This model combines S4D blocks with optional Mixture of Experts and demographic processing
    for improved EEG signal analysis and prediction.

    Args:
        n_chans: Number of EEG channels (default: 129)
        n_outputs: Number of output predictions (default: 1)
        n_times: Number of time points per sample (default: 200)
        d_model: Model hidden dimension (default: 128)
        n_layers: Number of S4D layers (default: 4)
        d_state: State dimension for S4D blocks (default: 64)
        bidirectional: Whether to use bidirectional S4D (default: True)
        n_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        use_moe: Whether to use Mixture of Experts (default: True)
        num_experts: Number of experts in MoE (default: 16)
        k: Number of experts to select (default: 4)
        use_demographics: Whether to use demographic information (default: False)
        demographic_dim: Dimension of demographic features (default: 5)
    """

    def __init__(self, n_chans: int = 129, n_outputs: int = 1, n_times: int = 200,
                 d_model: int = 128, n_layers: int = 4, d_state: int = 64,
                 bidirectional: bool = True, n_heads: int = 8, dropout: float = 0.1,
                 use_moe: bool = True, num_experts: int = 16, k: int = 4,
                 use_demographics: bool = False, demographic_dim: int = 5):
        super().__init__()

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(n_chans, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # S4D blocks
        self.s4d_blocks = nn.ModuleList()
        self.s4d_blocks.append(OptimizedS4DBlock(
            d_model=d_model,
            d_state=d_state,
            bidirectional=bidirectional,
            dropout=dropout
        ))

        block_dim = d_model * 2 if bidirectional else d_model
        for _ in range(1, n_layers):
            self.s4d_blocks.append(OptimizedS4DBlock(
                d_model=block_dim,
                d_state=d_state,
                bidirectional=False,
                dropout=dropout
            ))

        # Multi-head Attention Pooling
        pool_input_dim = d_model * 2 if bidirectional else d_model
        self.pooling = MultiHeadAttentionPooling(pool_input_dim, n_heads, dropout)

        # Demographic MLP (optional)
        self.use_demographics = use_demographics
        self.demographic_dim = demographic_dim

        if self.use_demographics:
            # Demographic processing branch
            self.demographic_mlp = nn.Sequential(
                nn.Linear(demographic_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(64),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.LayerNorm(256)
            )
            # Fusion dimension: pooled EEG features + demographic features
            fusion_dim = pool_input_dim + 256
        else:
            self.demographic_mlp = None
            fusion_dim = pool_input_dim

        # Mixture of Experts (if available)
        self.use_moe = use_moe and MOE_AVAILABLE
        if self.use_moe:
            # MoE: fusion_dim -> hidden_dim -> output_dim (512)
            self.moe = MoE(
                total_dim=fusion_dim,       # Input dimension (EEG + demographics if available)
                hidden_dim=256,             # Hidden dimension for experts
                out_dim=512,                # Output dimension
                num_experts=num_experts,
                k=k,
                num_modalities=2            # Keep default
            )
            head_input_dim = 512  # Output from MoE
        else:
            head_input_dim = fusion_dim

        # Main head with increased hidden units (512)
        self.main_head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_outputs)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, demographics: Optional[torch.Tensor] = None,
                return_entropy_loss: bool = False):
        """
        Forward pass through the model.

        Args:
            x: EEG data tensor of shape (batch, n_chans, n_times)
            demographics: Optional demographic features of shape (batch, demographic_dim)
            return_entropy_loss: Whether to return MoE entropy loss for regularization

        Returns:
            predictions: Output predictions of shape (batch, n_outputs)
            entropy_loss: (Optional) MoE entropy loss if return_entropy_loss is True
        """
        # Process EEG data
        # Input: (batch, n_chans=129, n_times=200)
        x = x.transpose(1, 2)  # -> (batch, 200, 129)
        x = self.input_projection(x)  # -> (batch, 200, 128)

        # Apply S4D blocks
        for block in self.s4d_blocks:
            x = block(x)

        # Pool temporal features
        x = self.pooling(x)  # -> (batch, 1, pool_input_dim)
        x = x.squeeze(1)  # -> (batch, pool_input_dim)

        # Process and fuse demographic features if available
        if self.use_demographics and demographics is not None:
            if demographics.dim() == 1:
                demographics = demographics.unsqueeze(0)  # Ensure batch dimension

            # Process demographics through MLP
            demo_features = self.demographic_mlp(demographics)  # -> (batch, 256)

            # Concatenate EEG and demographic features
            x = torch.cat([x, demo_features], dim=-1)  # -> (batch, fusion_dim)
        elif self.use_demographics and demographics is None:
            # If demographics expected but not provided, use zeros
            batch_size = x.shape[0]
            demo_zeros = torch.zeros(batch_size, self.demographic_dim, device=x.device)
            demo_features = self.demographic_mlp(demo_zeros)
            x = torch.cat([x, demo_features], dim=-1)

        # Apply MoE if available
        entropy_loss = None
        if self.use_moe:
            x, entropy_loss = self.moe(x)  # -> (batch, 512), entropy_loss

        # Final prediction
        x = self.main_head(x)  # -> (batch, n_outputs)

        if return_entropy_loss and entropy_loss is not None:
            return x, entropy_loss
        return x


# ====================================
# Submission Class (Required Format)
# ====================================

class Submission:
    """
    Submission class for EEG Challenge 2025
    Implements the required interface for model loading and inference
    """

    def __init__(self, SFREQ, DEVICE):
        """
        Initialize submission with sampling frequency and device

        Args:
            SFREQ: Sampling frequency (should be 100 Hz)
            DEVICE: torch device (cuda or cpu)
        """
        self.sfreq = SFREQ
        self.device = DEVICE
        self.n_chans = 129
        self.n_times = int(2 * SFREQ)  # 2 seconds of data

        # Model configuration (optimized based on experiments)
        self.model_config = {
            'n_chans': self.n_chans,
            'n_outputs': 1,
            'n_times': self.n_times,
            'd_model': 96,   # Reduced from 128
            'n_layers': 8,   # Increased for better capacity
            'd_state': 384,  # 4x d_model as recommended
            'bidirectional': True,
            'n_heads': 6,    # Reduced from 8
            'dropout': 0.0  # No dropout during inference
        }

    def get_model_challenge_1(self):
        """
        Load and return model for Challenge 1

        Returns:
            Trained model for Challenge 1
        """
        # Create model
        model = OptimizedS4DEEG(**self.model_config).to(self.device)

        # Load weights using resolve_path helper for final phase compatibility
        try:
            weights_path = resolve_path("weights_challenge_1.pt")
            # PyTorch 2.6 compatibility: explicitly set weights_only=False
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            print(f"[SUCCESS] Loaded weights for Challenge 1 from {weights_path}")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            print("   Using random initialization")
        except Exception as e:
            print(f"[WARNING] Could not load weights for Challenge 1: {e}")
            print("   Using random initialization")

        # Set to evaluation mode
        model.eval()
        return model

    def get_model_challenge_2(self):
        """
        Load and return model for Challenge 2

        Returns:
            Trained model for Challenge 2
        """
        # Create model (same architecture as Challenge 1)
        model = OptimizedS4DEEG(**self.model_config).to(self.device)

        # Load weights using resolve_path helper for final phase compatibility
        try:
            weights_path = resolve_path("weights_challenge_2.pt")
            # PyTorch 2.6 compatibility: explicitly set weights_only=False
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            print(f"[SUCCESS] Loaded weights for Challenge 2 from {weights_path}")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            print("   Using random initialization")
        except Exception as e:
            print(f"[WARNING] Could not load weights for Challenge 2: {e}")
            print("   Using random initialization")

        # Set to evaluation mode
        model.eval()
        return model


# ====================================
# Test the submission locally
# ====================================

if __name__ == "__main__":
    import time

    print("Testing Optimized S4D Submission...")
    print("="*60)

    # Test parameters
    SFREQ = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print(f"Sampling frequency: {SFREQ} Hz")

    # Create submission
    sub = Submission(SFREQ, DEVICE)

    # Test Challenge 1 model
    print("\nChallenge 1 Model:")
    model_1 = sub.get_model_challenge_1()
    model_1.eval()

    # Count parameters
    params = sum(p.numel() for p in model_1.parameters())
    print(f"  Parameters: {params:,}")

    # Test with dummy data
    batch_size = 32
    X = torch.randn(batch_size, 129, 200).to(DEVICE)

    # Test inference
    with torch.inference_mode():
        start = time.time()
        y_pred = model_1.forward(X)
        elapsed = time.time() - start

    print(f"  Output shape: {y_pred.shape}")
    print(f"  Inference time: {elapsed*1000:.2f} ms for batch of {batch_size}")
    print(f"  Per sample: {elapsed*1000/batch_size:.2f} ms")

    # Test Challenge 2 model
    print("\nChallenge 2 Model:")
    model_2 = sub.get_model_challenge_2()
    model_2.eval()

    with torch.inference_mode():
        y_pred = model_2.forward(X)

    print(f"  Output shape: {y_pred.shape}")
    print(f"  [SUCCESS] Model outputs correct shape (batch, 1)")

    print("\n" + "="*60)
    print("[SUCCESS] Submission test passed! Ready for competition.")
    print("="*60)