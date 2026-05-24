"""
HusformerBITIRMEEG - cross-modal transformer for EEG + Eye + Mouse frustration classification.

Architecture (Wang et al. IEEE TNNLS 2023, adapted):
  - EEG branch: pre-extracted 200-dim embeddings (no temporal encoder needed)
  - Eye branch: 1-D conv encoder over (6, 110) time series → 64-dim token
  - Mouse branch: 1-D conv encoder over (7, 210) time series → 64-dim token
  - Cross-modal attention: each modality queries the other two (symmetric)
  - Fusion MLP → binary classification (frustration / not)

Input shapes (per batch):
  eeg_emb:    (B, 200)   - from LaBraM fc_norm
  eye_ts:     (B, 6, 110) - 6 features × 110 time steps (50 Hz, 2200 ms)
  mouse_ts:   (B, 7, 210) - 7 features × 210 time steps (50 Hz, 4200 ms)

Target: ~80k total parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Modality encoders
# ---------------------------------------------------------------------------

class _TemporalEncoder(nn.Module):
    """1-D conv stack: (B, C_in, T) → (B, d_model)."""

    def __init__(self, in_channels: int, seq_len: int, d_model: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),   # → (B, 64, 1)
        )
        self.proj = nn.Linear(64, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        h = self.conv(x).squeeze(-1)   # (B, 64)
        return self.proj(h)            # (B, d_model)


class EEGEncoder(nn.Module):
    """Projects pre-extracted 200-dim LaBraM embeddings to d_model."""

    def __init__(self, in_dim: int = 200, d_model: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # (B, d_model)


# ---------------------------------------------------------------------------
# Cross-modal attention
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    One modality attends to the concatenation of the other two.
    Query: (B, 1, d_model); Key/Value: (B, 2, d_model) → (B, d_model).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                return_attn: bool = False):
        # query:   (B, d_model) → (B, 1, d_model)
        # context: list of (B, d_model) tensors → (B, K, d_model)
        q = query.unsqueeze(1)
        kv = torch.stack(context, dim=1)  # (B, K, d_model)
        out, attn_w = self.attn(q, kv, kv, need_weights=True,
                                average_attn_weights=True)  # (B, 1, d_model), (B, 1, K)
        refined = self.norm(query + out.squeeze(1))          # residual
        if return_attn:
            return refined, attn_w.squeeze(1)   # (B, K)
        return refined, None


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HusformerBITIRMEEG(nn.Module):
    """
    Cross-modal transformer for 3-modality frustration classification.

    Parameters
    ----------
    d_model : int
        Shared token dimension for all modalities (default 64).
    n_heads : int
        Attention heads in CrossModalAttention (default 4).
    n_classes : int
        Output classes (default 2: frustrated / not).
    dropout : float
        Dropout in classifier head.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_classes: int = 2,
        dropout: float = 0.3,
        # input dims
        eeg_dim: int = 200,
        eye_channels: int = 6,
        eye_len: int = 110,
        mouse_channels: int = 7,
        mouse_len: int = 210,
    ):
        super().__init__()

        # Modality encoders
        self.eeg_enc = EEGEncoder(eeg_dim, d_model)
        self.eye_enc = _TemporalEncoder(eye_channels, eye_len, d_model)
        self.mouse_enc = _TemporalEncoder(mouse_channels, mouse_len, d_model)

        # Cross-modal attention (each modality queries the other two)
        self.eeg_cross = CrossModalAttention(d_model, n_heads)
        self.eye_cross = CrossModalAttention(d_model, n_heads)
        self.mouse_cross = CrossModalAttention(d_model, n_heads)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(
        self,
        eeg_emb: torch.Tensor,    # (B, 200)
        eye_ts: torch.Tensor,     # (B, 6, 110)
        mouse_ts: torch.Tensor,   # (B, 7, 210)
        return_attn: bool = False,
    ):
        # Encode each modality to d_model
        e = self.eeg_enc(eeg_emb)      # (B, d_model)
        v = self.eye_enc(eye_ts)       # (B, d_model)
        m = self.mouse_enc(mouse_ts)   # (B, d_model)

        # Cross-modal attention: each modality queries the other two
        e2, eeg_attn = self.eeg_cross(e, [v, m], return_attn=return_attn)
        v2, eye_attn = self.eye_cross(v, [e, m], return_attn=return_attn)
        m2, mou_attn = self.mouse_cross(m, [e, v], return_attn=return_attn)

        # Concat and classify
        fused = torch.cat([e2, v2, m2], dim=-1)  # (B, 3*d_model)
        logits = self.classifier(fused)           # (B, n_classes)

        if return_attn:
            # eeg_attn[:, 0]=EEG→Eye, [:, 1]=EEG→Mouse etc.
            return logits, {
                "eeg": eeg_attn.cpu(),   # (B, 2)
                "eye": eye_attn.cpu(),   # (B, 2)
                "mouse": mou_attn.cpu(), # (B, 2)
            }
        return logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = HusformerBITIRMEEG()
    print(f"Total parameters: {model.n_params():,}")

    B = 8
    eeg = torch.randn(B, 200)
    eye = torch.randn(B, 6, 110)
    mou = torch.randn(B, 7, 210)

    out = model(eeg, eye, mou)
    print(f"Output shape: {out.shape}")   # expect (8, 2)
    assert out.shape == (B, 2)
    print("husformer smoke test PASSED")
