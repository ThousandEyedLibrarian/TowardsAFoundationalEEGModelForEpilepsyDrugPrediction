#!/usr/bin/env python3
"""
Improved Spatial-Aware S4D Model for EEG Challenge 2025.
Integrates actual S4D components from submission.py with spatial processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Import S4D components from submission.py
sys.path.append(str(Path(__file__).parent))
from submission import OptimizedS4DKernel, OptimizedS4DBlock


class SpatialChannelExtractor(nn.Module):
    """Advanced spatial feature extraction for EEG channels."""

    def __init__(self, n_chans=129, spatial_filters=32):
        super().__init__()
        # Separate processing for EEG channels (excluding reference)
        n_eeg_chans = n_chans - 1  # 128 EEG channels

        # Spatial convolution filters
        self.spatial_conv = nn.Conv2d(
            1, spatial_filters,
            kernel_size=(n_eeg_chans, 1),
            bias=False
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(spatial_filters)

        # Activation
        self.activation = nn.ELU()

        # Channel-wise attention mechanism
        self.channel_attention = nn.Sequential(
            nn.Linear(n_eeg_chans, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_eeg_chans),
            nn.Sigmoid()
        )

        # Reference channel processing
        self.ref_processor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, spatial_filters // 4)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels=129, time) - Raw EEG signal
        Returns:
            (batch, spatial_filters, time) - Spatial features
        """
        B, C, T = x.shape

        # Split EEG and reference channels
        eeg_channels = x[:, :128, :]  # (B, 128, T)
        ref_channel = x[:, 128:, :]   # (B, 1, T)

        # Apply channel attention
        channel_weights = self.channel_attention(
            eeg_channels.mean(dim=2)  # Average over time
        ).unsqueeze(2)  # (B, 128, 1)
        eeg_weighted = eeg_channels * channel_weights

        # Spatial filtering
        x_spatial = eeg_weighted.unsqueeze(1)  # (B, 1, 128, T)
        x_spatial = self.spatial_conv(x_spatial)  # (B, filters, 1, T)
        x_spatial = self.batch_norm(x_spatial)
        x_spatial = self.activation(x_spatial)
        x_spatial = x_spatial.squeeze(2)  # (B, filters, T)

        # Process reference channel separately
        ref_mean = ref_channel.mean(dim=2)  # (B, 1)
        ref_features = self.ref_processor(ref_mean)  # (B, filters/4)
        ref_features = ref_features.unsqueeze(2).expand(-1, -1, T)  # (B, filters/4, T)

        # Combine spatial and reference features
        # x_combined = torch.cat([x_spatial, ref_features], dim=1)
        # For simplicity, just return spatial features
        return x_spatial


class FrequencyDecomposition(nn.Module):
    """Decompose signal into frequency bands relevant for EEG."""

    def __init__(self, d_model=32, n_bands=5):
        super().__init__()
        # Define frequency bands (in samples at 100Hz)
        # Calculate channel distribution properly
        base_channels = d_model // n_bands
        remainder = d_model % n_bands
        channel_dims = [base_channels] * n_bands
        for i in range(remainder):
            channel_dims[i] += 1

        # Delta: 0.5-4Hz, Theta: 4-8Hz, Alpha: 8-13Hz, Beta: 13-30Hz, Gamma: 30-50Hz
        kernel_sizes = [200, 50, 25, 10, 5]
        self.band_kernels = nn.ModuleList([
            nn.Conv1d(d_model, channel_dims[i],
                     kernel_size=kernel_sizes[i], padding='same', groups=1)
            for i in range(n_bands)
        ])

        self.fusion = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels, time) - Frequency-aware features
        """
        bands = []
        for band_conv in self.band_kernels:
            band_features = band_conv(x)
            bands.append(band_features)

        x_freq = torch.cat(bands, dim=1)
        x_freq = self.fusion(x_freq)

        # Residual connection
        x = x + x_freq

        # Layer norm over channels
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, C, T)

        return x


class ResponseTimeAttention(nn.Module):
    """Attention mechanism focused on response time prediction."""

    def __init__(self, d_model, max_response_samples=150):
        super().__init__()
        # Focus on first 1.5 seconds (typical response window)
        self.max_response = max_response_samples

        # Learnable temporal importance weights
        self.temporal_weights = nn.Parameter(
            torch.ones(1, 1, 200) / 200
        )

        # Response prediction attention
        self.response_attention = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=25, padding='same'),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels, time) - Attention-weighted features
        """
        B, C, T = x.shape

        # Compute attention scores
        attention_scores = self.response_attention(x)  # (B, 1, T)

        # Apply temporal decay (early timepoints more important)
        temporal_mask = torch.sigmoid(self.temporal_weights[:, :, :T])
        attention_scores = attention_scores * temporal_mask

        # Normalize attention
        attention_scores = F.softmax(attention_scores, dim=2)

        # Apply attention
        x_attended = x * attention_scores

        return x_attended, attention_scores


class SpatialS4DEEG(nn.Module):
    """Complete Spatial-Aware S4D Model for EEG."""

    def __init__(self, n_chans=129, n_outputs=1, n_times=200,
                 spatial_filters=36, d_state=32, n_layers=2):
        super().__init__()

        # 1. Spatial feature extraction
        self.spatial_extractor = SpatialChannelExtractor(
            n_chans=n_chans,
            spatial_filters=spatial_filters
        )

        # 2. Frequency decomposition
        self.freq_decompose = FrequencyDecomposition(
            d_model=spatial_filters,
            n_bands=5
        )

        # 3. S4D blocks (using actual S4D from submission.py)
        self.s4d_blocks = nn.ModuleList()

        # First block: bidirectional
        self.s4d_blocks.append(OptimizedS4DBlock(
            d_model=spatial_filters,
            d_state=d_state,
            bidirectional=True,
            dropout=0.2
        ))

        # Subsequent blocks: unidirectional
        d_after_first = spatial_filters * 2  # Due to bidirectional
        for _ in range(1, n_layers):
            self.s4d_blocks.append(OptimizedS4DBlock(
                d_model=d_after_first,
                d_state=d_state,
                bidirectional=False,
                dropout=0.2
            ))

        # 4. Response time attention
        self.response_attention = ResponseTimeAttention(
            d_model=d_after_first,
            max_response_samples=150
        )

        # 5. Temporal pooling
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten()
        )

        # 6. Output prediction head
        hidden_dim = d_after_first * 8
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_outputs)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, channels=129, time=200) - Raw EEG signal
            return_attention: If True, also return attention weights
        Returns:
            prediction: (batch, 1) - Response time prediction
            attention: (batch, 1, time) - Attention weights (if requested)
        """
        # 1. Extract spatial features
        x = self.spatial_extractor(x)

        # 2. Frequency decomposition
        x = self.freq_decompose(x)

        # 3. S4D sequence modeling
        x = x.transpose(1, 2)  # (B, T, C) for S4D
        for block in self.s4d_blocks:
            x = block(x)
        x = x.transpose(1, 2)  # Back to (B, C, T)

        # 4. Apply response-time attention
        x, attention_weights = self.response_attention(x)

        # 5. Temporal pooling
        x = self.temporal_pool(x)

        # 6. Predict response time
        prediction = self.predictor(x)

        if return_attention:
            return prediction, attention_weights
        return prediction


def test_improved_model():
    """Test the improved spatial S4D model."""
    print("Testing Improved Spatial-S4D Model with Real S4D Components")
    print("=" * 60)

    # Create model
    model = SpatialS4DEEG(
        n_chans=129,
        n_outputs=1,
        n_times=200,
        spatial_filters=36,
        d_state=32,
        n_layers=2
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print(f"Original S4D: 453,729 parameters")
    print(f"Reduction: {(453729 - n_params) / 453729 * 100:.1f}%")

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 129, 200)

    model.eval()
    with torch.no_grad():
        y_pred, attention = model(x, return_attention=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Attention shape: {attention.shape}")

    # Test with real data if available
    data_path = Path("r5_full_data.npz")
    if data_path.exists():
        print("\n" + "=" * 60)
        print("Testing with Real Data")
        print("-" * 60)

        data = np.load(data_path)
        X = torch.tensor(data['X'][:32], dtype=torch.float32)
        y = torch.tensor(data['y'][:32], dtype=torch.float32)

        with torch.no_grad():
            y_pred = model(X)
            mse = F.mse_loss(y_pred, y)
            rmse = torch.sqrt(mse)

        print(f"Test RMSE (random weights): {rmse:.4f}")
        print(f"Mean prediction: {y_pred.mean():.3f}")
        print(f"Std prediction: {y_pred.std():.3f}")
        print(f"Target mean: {y.mean():.3f}")
        print(f"Target std: {y.std():.3f}")

    print("\n" + "=" * 60)
    print("Model Features:")
    print("-" * 60)
    print("1. Spatial channel extraction with attention")
    print("2. Frequency band decomposition (delta to gamma)")
    print("3. Real S4D blocks from submission.py")
    print("4. Response-time focused attention")
    print("5. Efficient architecture (~280K params)")
    print("=" * 60)

    return model


if __name__ == "__main__":
    # Test the improved model
    model = test_improved_model()

    # Save model
    torch.save(model.state_dict(), "spatial_s4d_improved.pt")
    print("\nModel weights saved to spatial_s4d_improved.pt")