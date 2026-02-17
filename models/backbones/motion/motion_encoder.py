"""Motion encoder wrapper for InternVideo2 Stage 2.

Supports two modes:
1. Raw motion input: body/hand motion tensors → CNN encoder → features
2. Token indices input: VQ-VAE tok_pose indices → codebook lookup → features

Both modes produce the same output format via fusion_proj.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Building blocks copied from BodyTokenize/src/model/vqvae.py
# to avoid a hard dependency on the BodyTokenize package.
# ============================================================

def _group_norm(num_channels: int, max_groups: int = 8):
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ResConv1DBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, drop: float = 0.0):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation)
        self.gn1 = _group_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation)
        self.gn2 = _group_norm(channels)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.gelu(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.drop(h)

        return F.gelu(x + h)


class CNNEncoder1D(nn.Module):
    """1-D CNN encoder: [B, T, Cin] -> [B, T', Cout], T' = T / temporal_compress."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_frames: int,
        temporal_compress: int,
        cnn_width: int = 256,
        cnn_depth: int = 8,
        cnn_kernel: int = 3,
        dilation_cycle: bool = True,
        dilation_max: int = 8,
        drop: float = 0.0,
        use_pos: bool = False,
        post_mlp: bool = False,
    ):
        super().__init__()
        self.r = temporal_compress

        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, cnn_width, kernel_size=self.r, stride=self.r),
            _group_norm(cnn_width),
            nn.GELU(),
        )

        blocks = []
        for i in range(cnn_depth):
            if dilation_cycle:
                d = 2 ** (i % int(math.log2(dilation_max) + 1))
                d = min(d, dilation_max)
            else:
                d = 1
            blocks.append(ResConv1DBlock(cnn_width, kernel=cnn_kernel, dilation=d, drop=drop))
        self.blocks = nn.Sequential(*blocks)

        self.pos = None
        self.proj = nn.Conv1d(cnn_width, out_dim, kernel_size=1)
        self.norm = None
        self.post_mlp = None

    def forward(self, x: torch.Tensor):
        # x: [B, T, Cin]
        x = x.permute(0, 2, 1)       # [B, Cin, T]
        x = self.stem(x)              # [B, width, T']
        x = self.blocks(x)            # [B, width, T']
        x = self.proj(x)              # [B, out_dim, T']
        x = x.permute(0, 2, 1)       # [B, T', out_dim]
        return x


# ============================================================
# Motion Encoder Wrapper
# ============================================================

class MotionEncoder(nn.Module):
    """Wraps body + hand encoders from BodyTokenize H2VQ for Stage 2.

    Supports two input modes:
    - forward(body_motion, hand_motion): raw motion → CNN encoder → features
    - forward_from_indices(idx): tok_pose indices → codebook lookup → features
    """

    def __init__(self, config):
        super().__init__()
        # Encoder hyperparameters (should match BodyTokenize ckpt)
        body_in_dim = config.get("body_in_dim", 263)
        hand_in_dim = config.get("hand_in_dim", 480)
        self.code_dim = config.get("code_dim", 256)
        self.body_tokens_per_t = config.get("body_tokens_per_t", 4)
        self.hand_tokens_per_t = config.get("hand_tokens_per_t", 4)
        self.tokens_per_t = self.body_tokens_per_t + self.hand_tokens_per_t  # 8
        num_frames = config.get("num_frames", 21)
        temporal_compress = config.get("temporal_compress", 1)
        cnn_width = config.get("cnn_width", 256)
        cnn_depth = config.get("cnn_depth", 8)
        cnn_dilation_max = config.get("cnn_dilation_max", 8)
        K = config.get("K", 1024)

        body_out_dim = self.body_tokens_per_t * self.code_dim  # 4*256 = 1024
        hand_out_dim = self.hand_tokens_per_t * self.code_dim  # 4*256 = 1024

        self.d_model = config.get("d_model", 768)

        # CNN encoders (for raw motion input)
        self.encB = CNNEncoder1D(
            in_dim=body_in_dim,
            out_dim=body_out_dim,
            num_frames=num_frames,
            temporal_compress=temporal_compress,
            cnn_width=cnn_width,
            cnn_depth=cnn_depth,
            cnn_dilation_max=cnn_dilation_max,
        )

        self.encH = CNNEncoder1D(
            in_dim=hand_in_dim,
            out_dim=hand_out_dim,
            num_frames=num_frames,
            temporal_compress=temporal_compress,
            cnn_width=cnn_width,
            cnn_depth=cnn_depth,
            cnn_dilation_max=cnn_dilation_max,
        )

        # Codebook buffers (for token index input)
        self.register_buffer("codebook_B", torch.zeros(K, self.code_dim))
        self.register_buffer("codebook_H", torch.zeros(K, self.code_dim))

        # Fuse body + hand -> d_model
        self.fusion_proj = nn.Sequential(
            nn.Linear(body_out_dim + hand_out_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def forward(self, body_motion: torch.Tensor, hand_motion: torch.Tensor):
        """Raw motion input mode.

        Args:
            body_motion: [B, T, 263]
            hand_motion: [B, T, 480]
        Returns:
            motion_embeds: [B, T', d_model], pooled: [B, d_model]
        """
        zB = self.encB(body_motion)   # [B, T', 1024]
        zH = self.encH(hand_motion)   # [B, T', 1024]
        z = torch.cat([zB, zH], dim=-1)  # [B, T', 2048]
        z = self.fusion_proj(z)        # [B, T', d_model]

        pooled = z.mean(dim=1)         # [B, d_model]
        return z, pooled

    def forward_from_indices(self, idx: torch.Tensor):
        """Token index input mode — skips CNN encoder, uses codebook lookup.

        Args:
            idx: [B, N] flat token indices where N = T * (body_tokens + hand_tokens).
                 Layout per timestep: [B0,B1,B2,B3, H0,H1,H2,H3] (body then hand).
        Returns:
            motion_embeds: [B, T, d_model], pooled: [B, d_model]
        """
        B = idx.shape[0]
        idx = idx.reshape(B, -1, self.tokens_per_t)  # [B, T, 8]
        idxB = idx[:, :, :self.body_tokens_per_t]     # [B, T, 4]
        idxH = idx[:, :, self.body_tokens_per_t:]     # [B, T, 4]

        # Codebook lookup
        embB = self.codebook_B[idxB]  # [B, T, 4, 256]
        embH = self.codebook_H[idxH]  # [B, T, 4, 256]

        T = idx.shape[1]
        zB = embB.reshape(B, T, self.body_tokens_per_t * self.code_dim)   # [B, T, 1024]
        zH = embH.reshape(B, T, self.hand_tokens_per_t * self.code_dim)   # [B, T, 1024]

        z = torch.cat([zB, zH], dim=-1)  # [B, T, 2048]
        z = self.fusion_proj(z)           # [B, T, d_model]

        pooled = z.mean(dim=1)            # [B, d_model]
        return z, pooled

    @classmethod
    def from_pretrained(cls, config):
        """Build MotionEncoder and load weights from BodyTokenize VQ-VAE ckpt."""
        model = cls(config)

        ckpt_path = config.get("ckpt_path", None)
        if ckpt_path is None:
            logger.warning("No motion encoder checkpoint provided, using random init.")
            return model

        logger.info(f"Loading motion encoder weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)

        # Extract encoder weights
        encB_state = {}
        encH_state = {}
        for k, v in state_dict.items():
            if k.startswith("encB."):
                encB_state[k.replace("encB.", "")] = v
            elif k.startswith("encH."):
                encH_state[k.replace("encH.", "")] = v

        msg_b = model.encB.load_state_dict(encB_state, strict=False)
        msg_h = model.encH.load_state_dict(encH_state, strict=False)
        logger.info(f"encB load: {msg_b}")
        logger.info(f"encH load: {msg_h}")

        # Extract codebook weights
        if "qB.codebook" in state_dict:
            model.codebook_B.copy_(state_dict["qB.codebook"])
            logger.info(f"Loaded codebook_B: {model.codebook_B.shape}")
        else:
            logger.warning("qB.codebook not found in checkpoint!")

        if "qH.codebook" in state_dict:
            model.codebook_H.copy_(state_dict["qH.codebook"])
            logger.info(f"Loaded codebook_H: {model.codebook_H.shape}")
        else:
            logger.warning("qH.codebook not found in checkpoint!")

        return model
