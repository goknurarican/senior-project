"""
LaBraM wrapper for BITIRMEEG Approach A.

Loads the pretrained LaBraM-Base backbone (12-layer transformer, 200-dim embeddings)
and exposes a get_embeddings() method that takes raw EEG epochs and returns
(batch, 200) feature vectors via a hook on the fc_norm layer.

Pretrained weight compatibility:
  - 145/221 keys loaded (all 12 transformer block attn/MLP weights - 99.4% of params)
  - position_embedding and temporal_embedding randomly initialized (32ch vs 128ch pretrained)
  - gamma_1/2, q_norm/k_norm skipped (braindecode 1.2 vs 1.3 version mismatch)
"""

import logging
from pathlib import Path

import numpy as np
import torch
from braindecode.models import Labram
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

PRETRAINED_PATH = Path(__file__).parents[1] / "models" / "pytorch_model.bin"
N_CHANS = 32
N_TIMES = 400       # 2 s at 200 Hz
SFREQ_TARGET = 200  # Hz


def _load_compatible_weights(model: Labram, ckpt_path: Path) -> dict:
    """Load pretrained weights, skipping keys whose shapes don't match."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_sd = model.state_dict()

    loaded, skipped = [], []
    filtered = {}
    for k, v in ckpt.items():
        if k not in model_sd:
            skipped.append((k, "not_in_model"))
            continue
        if v.shape != model_sd[k].shape:
            skipped.append((k, f"shape {tuple(v.shape)} vs {tuple(model_sd[k].shape)}"))
            continue
        filtered[k] = v
        loaded.append(k)

    model_sd.update(filtered)
    model.load_state_dict(model_sd)

    log.info(f"LaBraM pretrained: loaded {len(loaded)}/{len(ckpt)} keys")
    log.info(f"  Skipped {len(skipped)} keys (shape mismatch or absent)")
    # Confirm transformer blocks
    blocks_loaded = [k for k in loaded if k.startswith("blocks.")]
    log.info(f"  Transformer block keys loaded: {len(blocks_loaded)}")
    return {"loaded": loaded, "skipped": skipped}


def load_labram_base(pretrained_path: Path = PRETRAINED_PATH,
                     device: str = "cpu") -> "LabramEncoder":
    """Return a LabramEncoder with pretrained weights and fc_norm hook installed."""
    encoder = LabramEncoder(device=device)
    info = _load_compatible_weights(encoder.model, pretrained_path)
    return encoder


class LabramEncoder:
    """
    Wraps Labram and extracts embeddings from the fc_norm layer.

    Usage:
        encoder = load_labram_base()
        emb = encoder.get_embeddings(epochs_np)  # (N, 32, n_times_500Hz) → (N, 200)
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = Labram(
            n_chans=N_CHANS,
            n_times=N_TIMES,
            sfreq=SFREQ_TARGET,
            n_outputs=2,
            n_layers=12,
            emb_size=200,
            att_num_heads=10,
        ).to(self.device)
        self.model.eval()
        self._embedding: torch.Tensor | None = None
        self._hook_handle = self._install_hook()

    def _install_hook(self):
        def hook_fn(module, input, output):
            # fc_norm output: (batch, 200) - already pooled by use_mean_pooling=True
            self._embedding = output.detach()

        handle = self.model.fc_norm.register_forward_hook(hook_fn)
        return handle

    def _preprocess(self, epochs_np: np.ndarray) -> torch.Tensor:
        """
        epochs_np: (N, 32, n_times) - raw ERP epochs at any sfreq.
        Steps:
          1. Resample to SFREQ_TARGET (200 Hz) using polyphase if needed
          2. Take first N_TIMES (400) samples
          3. Z-score per epoch per channel
        Returns (N, 32, 400) float32 tensor.
        """
        N, C, T = epochs_np.shape

        # Resample if necessary (assumes input is 500 Hz from EEG pipeline)
        # resample_poly(up, down) - 200/500 = 2/5
        if T != N_TIMES:
            # Attempt to detect sfreq from n_times: ERP window is -200ms to +2000ms = 2200ms
            # at 500 Hz → 1101 samples; at 200 Hz → 440 samples
            if T >= 440:
                up, down = 2, 5  # 500→200 Hz
                resampled = np.zeros((N, C, 440), dtype=np.float32)
                for i in range(N):
                    for c in range(C):
                        resampled[i, c] = resample_poly(epochs_np[i, c], up, down)[:440]
                epochs_np = resampled
            # Take first N_TIMES samples
            epochs_np = epochs_np[:, :, :N_TIMES]

        # Z-score per epoch, per channel
        mean = epochs_np.mean(axis=-1, keepdims=True)
        std = epochs_np.std(axis=-1, keepdims=True) + 1e-8
        epochs_np = (epochs_np - mean) / std

        return torch.tensor(epochs_np, dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def get_embeddings(self, epochs_np: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extract 200-dim embeddings for all epochs.

        Args:
            epochs_np: (N, 32, n_times) float array - ERP epochs
            batch_size: inference batch size

        Returns:
            embeddings: (N, 200) float32 numpy array
        """
        x = self._preprocess(epochs_np)
        N = x.shape[0]
        all_embs = []

        for start in range(0, N, batch_size):
            batch = x[start:start + batch_size]
            self._embedding = None
            _ = self.model(batch)
            if self._embedding is None:
                raise RuntimeError("fc_norm hook did not fire - check model structure")
            all_embs.append(self._embedding.cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    def remove_hook(self):
        self._hook_handle.remove()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("Loading LaBraM encoder...")
    encoder = load_labram_base()

    # Smoke test: random epoch (32 channels, 1101 samples at 500 Hz)
    dummy = np.random.randn(4, 32, 1101).astype(np.float32)
    emb = encoder.get_embeddings(dummy)
    log.info(f"Embedding shape: {emb.shape}")   # expect (4, 200)
    assert emb.shape == (4, 200), f"Unexpected shape: {emb.shape}"
    log.info("labram_wrapper smoke test PASSED")
