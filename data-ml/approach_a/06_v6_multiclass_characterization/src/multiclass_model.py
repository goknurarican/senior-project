"""
V6 Multi-Class Scenario Classifier

Predicts: which scenario (or control) does this epoch represent?
15 classes: control + 14 frustration scenarios

Architecture:
- EEG branch: 25 oscillation time series -> small transformer encoder
- Eye branch: 1D-CNN
- Mouse branch: 1D-CNN
- Cross-modal attention (Husformer-style)
- 15-class classifier head

Total parameters: ~80k
"""

import torch
import torch.nn as nn


class OscillationTransformerEncoder(nn.Module):
    """
    Each oscillation time series is treated as a token in a sequence.
    Transformer learns inter-feature interactions.
    """
    def __init__(self, n_features=25, n_timepoints=110, hidden=64, n_heads=4, n_layers=2):
        super().__init__()
        self.feature_proj = nn.Conv1d(1, hidden, kernel_size=5, padding=2)
        self.pos_encoding  = nn.Parameter(torch.randn(1, n_timepoints, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=128,
            dropout=0.3, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden, 16), nn.Tanh(), nn.Linear(16, 1)
        )
        self.n_features = n_features
        self.hidden = hidden

    def forward(self, x):
        #x: (batch, 25, 110)
        batch_size = x.size(0)
        feature_outputs = []
        for i in range(self.n_features):
            f_x   = x[:, i:i+1, :]                          # (batch, 1, 110)
            f_emb = self.feature_proj(f_x).transpose(1, 2)  # (batch, 110, hidden)
            f_emb = f_emb + self.pos_encoding
            f_out = self.transformer(f_emb)                  # (batch, 110, hidden)
            feature_outputs.append(f_out.mean(dim=1))        # (batch, hidden)
        stacked     = torch.stack(feature_outputs, dim=1)    # (batch, 25, hidden)
        attn_logits = self.feature_attention(stacked).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=-1)    # (batch, 25)
        output = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return output, attn_weights


class EyeBranch(nn.Module):
    def __init__(self, n_features=6, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
            nn.Flatten(),
            nn.Linear(64 * 10, hidden)
        )
    def forward(self, x): return self.encoder(x)


class MouseBranch(nn.Module):
    def __init__(self, n_features=7, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
            nn.Flatten(),
            nn.Linear(64 * 10, hidden)
        )
    def forward(self, x): return self.encoder(x)


class CrossModalAttention(nn.Module):
    def __init__(self, hidden=64, n_heads=4):
        super().__init__()
        self.attn_eeg   = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.attn_eye   = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.attn_mouse = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, eeg, eye, mouse):
        modalities = torch.stack([eeg, eye, mouse], dim=1)
        eeg_out,   eeg_w   = self.attn_eeg(eeg.unsqueeze(1),   modalities, modalities)
        eye_out,   eye_w   = self.attn_eye(eye.unsqueeze(1),   modalities, modalities)
        mouse_out, mouse_w = self.attn_mouse(mouse.unsqueeze(1), modalities, modalities)
        return (
            self.norm(eeg_out.squeeze(1)   + eeg),
            self.norm(eye_out.squeeze(1)   + eye),
            self.norm(mouse_out.squeeze(1) + mouse),
            {'eeg': eeg_w.squeeze(1), 'eye': eye_w.squeeze(1), 'mouse': mouse_w.squeeze(1)}
        )


class V6MultiClassModel(nn.Module):
    """
    15-class classifier with two levels of interpretability:
    - Feature attention (which oscillation feature is important)
    - Modality attention (which modality drove decision)
    """
    def __init__(self, n_classes=15, hidden=64, n_heads=4, dropout=0.3):
        super().__init__()
        self.eeg_encoder  = OscillationTransformerEncoder(
            n_features=25, n_timepoints=110, hidden=hidden, n_heads=n_heads
        )
        self.eye_branch   = EyeBranch(n_features=6, hidden=hidden)
        self.mouse_branch = MouseBranch(n_features=7, hidden=hidden)
        self.cross_modal  = CrossModalAttention(hidden, n_heads)
        self.classifier   = nn.Sequential(
            nn.Linear(hidden * 3, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        self.aux_eeg   = nn.Linear(hidden, n_classes)
        self.aux_eye   = nn.Linear(hidden, n_classes)
        self.aux_mouse = nn.Linear(hidden, n_classes)

    def forward(self, osc_ts, eye_ts, mouse_ts, modality_mask=None):
        eeg, feature_attn = self.eeg_encoder(osc_ts)
        eye   = self.eye_branch(eye_ts)
        mouse = self.mouse_branch(mouse_ts)

        if modality_mask is not None and self.training:
            if not modality_mask.get('eeg', True):
                eeg   = torch.zeros_like(eeg)
            if not modality_mask.get('eye', True):
                eye   = torch.zeros_like(eye)
            if not modality_mask.get('mouse', True):
                mouse = torch.zeros_like(mouse)

        eeg_a, eye_a, mouse_a, modal_attn = self.cross_modal(eeg, eye, mouse)
        fused  = torch.cat([eeg_a, eye_a, mouse_a], dim=-1)
        logits = self.classifier(fused)

        aux_logits = {
            'eeg':   self.aux_eeg(eeg),
            'eye':   self.aux_eye(eye),
            'mouse': self.aux_mouse(mouse),
        }
        return logits, aux_logits, {
            'feature_attention': feature_attn,
            'modality_attention': modal_attn,
        }


if __name__ == "__main__":
    model = V6MultiClassModel(n_classes=15)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    #Smoke test
    osc   = torch.randn(4, 25, 110)
    eye   = torch.randn(4, 6, 110)
    mouse = torch.randn(4, 7, 210)
    logits, aux, attn = model(osc, eye, mouse)
    print(f"Logits shape: {logits.shape}")     # (4, 15)
    print(f"Feature attn: {attn['feature_attention'].shape}")  # (4, 25)
