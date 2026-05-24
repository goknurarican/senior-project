"""
V5: Oscillation-Guided LaBraM Adapter + Balanced Multimodal Fusion

Key innovations:
  1. LaBraM frozen (via pre-computed embeddings), adapter distills to interpretable 6-dim space
  2. OscillationGuidedAdapter: 200-dim LaBraM → 6-dim oscillation-aligned projection
  3. OscillationTemporalEncoder: raw osc time series → band attention
  4. Modality dropout (prevents EEG dominance)
  5. Auxiliary modality losses (forces each branch to learn independently)
  6. Cross-modal attention (Husformer-style)

References:
  LaBraM: Jiang et al. (2024), ICLR Spotlight
  Knowledge distillation: Hinton et al. (2015)
  Modality dropout: Neverova et al. (2016), TPAMI
  Cross-modal attention: Wang et al. (2023), IEEE TNNLS
  Oscillation features: Cavanagh & Frank (2014), Trends Cog Sci
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OscillationGuidedAdapter(nn.Module):
    """
    Projects LaBraM 200-dim embedding to 6-dim interpretable oscillation space.
    Anchor alignment loss during training pushes this projection to correlate
    with the actual oscillation temporal mean (knowledge distillation from literature).
    """
    def __init__(self, labram_dim=200, n_bands=6, dropout=0.3):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(labram_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_bands)
        )

    def forward(self, labram_emb):
        # labram_emb: (B, 200) → (B, 6)
        return self.adapter(labram_emb)


class OscillationTemporalEncoder(nn.Module):
    """
    Per-band 1D-CNN encoder over raw oscillation time series.
    Produces band attention weights (interpretable) + fused representation.
    """
    def __init__(self, n_bands=6, n_timepoints=110, hidden=32):
        super().__init__()
        self.n_bands = n_bands
        per_band = max(4, hidden // n_bands)   # 5 for hidden=32, n_bands=6 → use 5
        self.per_band = per_band

        self.band_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(10),   # 110/10=11 ✓ divisible on MPS
                nn.Flatten(),
                nn.Linear(32 * 10, per_band)
            ) for _ in range(n_bands)
        ])

        # Band attention for interpretability
        self.band_attention = nn.Sequential(
            nn.Linear(per_band, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Project concatenated per-band features → hidden
        self.proj = nn.Linear(n_bands * per_band, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: (B, 6, 110)
        band_feats = []
        for i, enc in enumerate(self.band_encoders):
            band_feats.append(enc(x[:, i:i+1, :]))   # (B, per_band)
        stacked = torch.stack(band_feats, dim=1)       # (B, 6, per_band)

        # Band attention weights
        attn_logits = self.band_attention(stacked).squeeze(-1)  # (B, 6)
        band_attn   = F.softmax(attn_logits, dim=-1)            # (B, 6)

        # Weighted-attended summary (for aux use) + full concat → proj
        full   = stacked.reshape(stacked.size(0), -1)  # (B, 6*per_band)
        output = self.norm(self.proj(full))              # (B, hidden)
        return output, band_attn


class EyeBranch(nn.Module):
    def __init__(self, n_features=6, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
            nn.Flatten(),
            nn.Linear(32 * 10, hidden)
        )

    def forward(self, x):
        return self.encoder(x)   # (B, hidden)


class MouseBranch(nn.Module):
    def __init__(self, n_features=7, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
            nn.Flatten(),
            nn.Linear(32 * 10, hidden)
        )

    def forward(self, x):
        return self.encoder(x)   # (B, hidden)


class CrossModalAttention(nn.Module):
    """Husformer-style: each modality attends to the other two."""
    def __init__(self, hidden=32, n_heads=4):
        super().__init__()
        self.attn_eeg   = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.attn_eye   = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.attn_mouse = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, eeg, eye, mouse):
        # Stack all three as context
        ctx = torch.stack([eeg, eye, mouse], dim=1)   # (B, 3, H)

        eeg_out,   eeg_w   = self.attn_eeg(eeg.unsqueeze(1),   ctx, ctx)
        eye_out,   eye_w   = self.attn_eye(eye.unsqueeze(1),   ctx, ctx)
        mouse_out, mouse_w = self.attn_mouse(mouse.unsqueeze(1), ctx, ctx)

        eeg_r   = self.norm(eeg_out.squeeze(1)   + eeg)
        eye_r   = self.norm(eye_out.squeeze(1)   + eye)
        mouse_r = self.norm(mouse_out.squeeze(1) + mouse)

        attn_weights = {
            'eeg':   eeg_w.squeeze(1),    # (B, 3)
            'eye':   eye_w.squeeze(1),
            'mouse': mouse_w.squeeze(1),
        }
        return eeg_r, eye_r, mouse_r, attn_weights


class HybridV5Model(nn.Module):
    """
    Full V5 model.

    Inputs per forward pass:
      labram_emb : (B, 200)   frozen LaBraM embeddings
      osc_ts     : (B, 6, 110)  oscillation time series
      eye_ts     : (B, 6, 110)
      mouse_ts   : (B, 7, 210)
      modality_mask : {'eeg': bool, 'eye': bool, 'mouse': bool} - for dropout

    Returns:
      logits      : (B, 2)
      aux_logits  : dict  {'eeg', 'eye', 'mouse'} → (B, 2)
      attn_info   : dict with 'band_attention', 'modality_attention', 'adapter_output'
    """
    def __init__(self, hidden=32, n_heads=4, n_classes=2, dropout=0.3):
        super().__init__()

        # EEG path A: adapter  200 → 6
        self.labram_adapter = OscillationGuidedAdapter(
            labram_dim=200, n_bands=6, dropout=dropout
        )
        # EEG path B: temporal  (6,110) → hidden
        self.eeg_temporal = OscillationTemporalEncoder(
            n_bands=6, n_timepoints=110, hidden=hidden
        )
        # Combine A (6) + B (hidden=32) → hidden
        self.eeg_combine = nn.Sequential(
            nn.Linear(6 + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.eye_branch   = EyeBranch(n_features=6, hidden=hidden)
        self.mouse_branch = MouseBranch(n_features=7, hidden=hidden)

        self.cross_modal = CrossModalAttention(hidden, n_heads)

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # Auxiliary per-modality classifiers
        self.aux_eeg_clf   = nn.Linear(hidden, n_classes)
        self.aux_eye_clf   = nn.Linear(hidden, n_classes)
        self.aux_mouse_clf = nn.Linear(hidden, n_classes)

    def forward(self, labram_emb, osc_ts, eye_ts, mouse_ts, modality_mask=None):
        # EEG
        adapter_out = self.labram_adapter(labram_emb)           # (B, 6)
        temporal_out, band_attn = self.eeg_temporal(osc_ts)     # (B, 32), (B, 6)
        eeg = self.eeg_combine(torch.cat([adapter_out, temporal_out], dim=-1))  # (B, 32)

        eye   = self.eye_branch(eye_ts)       # (B, 32)
        mouse = self.mouse_branch(mouse_ts)   # (B, 32)

        # Modality dropout (applied to entire batch for chosen modality)
        if modality_mask is not None and self.training:
            if not modality_mask.get('eeg', True):
                eeg   = torch.zeros_like(eeg)
            if not modality_mask.get('eye', True):
                eye   = torch.zeros_like(eye)
            if not modality_mask.get('mouse', True):
                mouse = torch.zeros_like(mouse)

        # Cross-modal attention
        eeg_r, eye_r, mouse_r, mod_attn = self.cross_modal(eeg, eye, mouse)

        # Main logits
        fused  = torch.cat([eeg_r, eye_r, mouse_r], dim=-1)  # (B, 96)
        logits = self.classifier(fused)

        # Auxiliary logits (pre-attention representations)
        aux_logits = {
            'eeg':   self.aux_eeg_clf(eeg),
            'eye':   self.aux_eye_clf(eye),
            'mouse': self.aux_mouse_clf(mouse),
        }

        attn_info = {
            'band_attention':    band_attn,     # (B, 6)
            'modality_attention': mod_attn,     # dict of (B, 3)
            'adapter_output':    adapter_out,   # (B, 6) interpretable
        }
        return logits, aux_logits, attn_info


def count_parameters(model):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = HybridV5Model()
    total, trainable = count_parameters(model)
    print(f"Total params   : {total:,}")
    print(f"Trainable params: {trainable:,}")
    # Quick forward test
    B = 4
    lab = torch.randn(B, 200)
    osc = torch.randn(B, 6, 110)
    eye = torch.randn(B, 6, 110)
    mou = torch.randn(B, 7, 210)
    logits, aux, attn = model(lab, osc, eye, mou)
    print(f"logits: {logits.shape}")
    print(f"aux_eeg: {aux['eeg'].shape}")
    print(f"band_attn: {attn['band_attention'].shape}")
    print(f"adapter_out: {attn['adapter_output'].shape}")
