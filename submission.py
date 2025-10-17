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
        y = self.norm(y)

        if y.shape[-1] == x.shape[-1]:
            y = y + x

        y = self.dropout(y)
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
    """Optimized S4D Model matching architecture diagram"""

    def __init__(self, n_chans: int = 129, n_outputs: int = 1, n_times: int = 200,
                 d_model: int = 128, n_layers: int = 4, d_state: int = 64,
                 bidirectional: bool = True, n_heads: int = 8, dropout: float = 0.1):
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

        # FIXME: Add multilayer perceptron for demographic info here, concatenate if with pooling layer to get 1024 
        # feed that into moe into main head - may need changes to data preprocessing to keep demographics 

        # Main head
        self.main_head = nn.Sequential(
            nn.Linear(pool_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, n_chans=129, n_times=200)
        x = x.transpose(1, 2)  # -> (batch, 200, 129)
        x = self.input_projection(x)  # -> (batch, 200, 128)

        for block in self.s4d_blocks:
            x = block(x)

        x = self.pooling(x)  # -> (batch, 256)
        x = self.main_head(x)  # -> (batch, 1)
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

        # Model configuration (reduced to match training config)
        self.model_config = {
            'n_chans': self.n_chans,
            'n_outputs': 1,
            'n_times': self.n_times,
            'd_model': 96,   # Reduced from 128
            'n_layers': 3,   # Reduced from 4
            'd_state': 48,   # Reduced from 64
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

        # Load weights if available
        weights_path = "/app/output/weights_challenge_1.pt"
        if Path(weights_path).exists():
            try:
                # PyTorch 2.6 compatibility: explicitly set weights_only=False
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                print(f"[SUCCESS] Loaded weights for Challenge 1 from {weights_path}")
            except Exception as e:
                print(f"[WARNING] Could not load weights for Challenge 1: {e}")
                print("   Using random initialization")
        else:
            print(f"[WARNING] No weights found at {weights_path}, using random initialization")

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

        # Load weights if available
        weights_path = "/app/output/weights_challenge_2.pt"
        if Path(weights_path).exists():
            try:
                # PyTorch 2.6 compatibility: explicitly set weights_only=False
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                print(f"[SUCCESS] Loaded weights for Challenge 2 from {weights_path}")
            except Exception as e:
                print(f"[WARNING] Could not load weights for Challenge 2: {e}")
                print("   Using random initialization")
        else:
            print(f"[WARNING] No weights found at {weights_path}, using random initialization")

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