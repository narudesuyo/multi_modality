"""Motion encoder wrapper for InternVideo2 Stage 2.

Loads the BodyTokenize H2VQ model via build_model_from_args and extracts
encoders + codebooks for downstream use.

Supports two modes:
1. Raw motion input: body/hand motion tensors → CNN encoder → features
2. Token indices input: VQ-VAE tok_pose indices → codebook lookup → features
"""

import logging
import re
import sys
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _extract_omegaconf_dict(obj):
    """Extract a plain dict from an OmegaConf DictConfig (possibly broken by pickle).

    torch.load may deserialize DictConfig with broken iteration.
    We access _content directly and unwrap ValueNode objects.
    """
    if isinstance(obj, dict):
        return obj

    # Access internal _content dict
    content = None
    if hasattr(obj, '_content') and isinstance(obj._content, dict):
        content = obj._content
    elif hasattr(obj, '__dict__'):
        d = obj.__dict__
        if '_content' in d and isinstance(d['_content'], dict):
            content = d['_content']

    if content is None:
        logger.warning(f"Cannot extract dict from {type(obj)}, returning empty")
        return {}

    # Unwrap ValueNode objects in _content
    result = {}
    for k, v in content.items():
        val = _unwrap_value_node(v)
        if val is not None:
            result[k] = val

    # Resolve ${...} interpolation references
    for k, v in list(result.items()):
        if isinstance(v, str) and '${' in v:
            refs = re.findall(r'\$\{(\w+)\}', v)
            resolved = v
            for ref in refs:
                if ref in result:
                    resolved = resolved.replace(f'${{{ref}}}', str(result[ref]))
            result[k] = resolved

    # Convert string bools/ints/floats
    for k, v in list(result.items()):
        if isinstance(v, str):
            vl = v.lower()
            if vl in ('true', 'false'):
                result[k] = vl == 'true'
            else:
                try:
                    result[k] = int(v)
                except ValueError:
                    try:
                        result[k] = float(v)
                    except ValueError:
                        pass

    return result


def _unwrap_value_node(v):
    """Recursively unwrap OmegaConf ValueNode to plain Python value."""
    # Plain Python type
    if isinstance(v, (int, float, bool, str, type(None))):
        return v
    # ValueNode with _value() method
    if hasattr(v, '_value'):
        try:
            return v._value()
        except Exception:
            pass
    # ValueNode with _val attribute
    if hasattr(v, '_val'):
        return v._val
    # Try str() as last resort
    try:
        return str(v)
    except Exception:
        return None


class MotionEncoder(nn.Module):
    """Wraps body + hand encoders from BodyTokenize H2VQ for Stage 2.

    Supports two input modes:
    - forward(body_motion, hand_motion): raw motion → CNN encoder → features
    - forward_from_indices(idx): tok_pose indices → codebook lookup → features
    """

    def __init__(self, h2vq_model, d_model=768):
        super().__init__()

        self.h2vq = h2vq_model
        self.code_dim = h2vq_model.code_dim
        self.body_tokens_per_t = h2vq_model.body_tokens_per_t
        self.hand_tokens_per_t = h2vq_model.hand_tokens_per_t
        self.tokens_per_t = self.body_tokens_per_t + self.hand_tokens_per_t

        # Expose encB, encH, codebooks as attributes for freezing / saving
        self.encB = h2vq_model.encB
        self.encH = h2vq_model.encH
        self.codebook_B = h2vq_model.qB.codebook
        self.codebook_H = h2vq_model.qH.codebook

        body_out_dim = self.body_tokens_per_t * self.code_dim
        hand_out_dim = self.hand_tokens_per_t * self.code_dim

        self.d_model = d_model

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
        zB = self.encB(body_motion)   # [B, T', body_out]
        zH = self.encH(hand_motion)   # [B, T', hand_out]
        z = torch.cat([zB, zH], dim=-1)
        z = self.fusion_proj(z)

        pooled = z.mean(dim=1)
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
        zB = embB.reshape(B, T, self.body_tokens_per_t * self.code_dim)
        zH = embH.reshape(B, T, self.hand_tokens_per_t * self.code_dim)

        z = torch.cat([zB, zH], dim=-1)
        # Cast to fusion_proj dtype to fix mat1/mat2 mismatch under DeepSpeed bf16
        # (codebook buffers are float32, fusion_proj may be bfloat16)
        proj_dtype = next(self.fusion_proj.parameters()).dtype
        z = z.to(proj_dtype)
        z = self.fusion_proj(z)

        pooled = z.mean(dim=1)
        return z, pooled

    @classmethod
    def from_pretrained(cls, config, args=None):
        """Build MotionEncoder by loading H2VQ via BodyTokenize's build_model_from_args."""
        ckpt_path = config.get("ckpt_path", None)
        if ckpt_path is None:
            raise ValueError("ckpt_path is required for MotionEncoder.from_pretrained")

        vqvae_config_path = config.get("vqvae_config", None)

        logger.info(f"Loading H2VQ checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Add BodyTokenize to sys.path so we can import it
        # ckpt_path is like /path/to/BodyTokenize/ckpt_vq/ckpt_best.pt
        body_tokenize_root = os.path.dirname(os.path.dirname(ckpt_path))
        if body_tokenize_root not in sys.path:
            sys.path.insert(0, body_tokenize_root)
            logger.info(f"Added {body_tokenize_root} to sys.path")

        from src.train.utils import build_model_from_args, _safe_merge_args_from_ckpt

        # Build args for build_model_from_args: prefer vqvae_config yaml, fallback to ckpt args.
        if vqvae_config_path and os.path.isfile(vqvae_config_path):
            logger.info(f"Loading VQVAE config from {vqvae_config_path}")
            h2vq_args = OmegaConf.load(vqvae_config_path)
            h2vq_args = _safe_merge_args_from_ckpt(h2vq_args, ckpt)
        else:
            ckpt_args_raw = ckpt.get("args", {})
            ckpt_args = _extract_omegaconf_dict(ckpt_args_raw)
            logger.info(f"Using args from checkpoint (vqvae_config not provided): {list(ckpt_args.keys())}")
            h2vq_args = OmegaConf.create(ckpt_args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h2vq = build_model_from_args(h2vq_args, device)

        # logger.info(f"H2VQ constructed: T={ckpt_args.get('T')}, include_fingertips={include_fingertips}")

        # Load weights
        state_dict = ckpt.get("model", ckpt)
        msg = h2vq.load_state_dict(state_dict, strict=False)
        logger.info(f"H2VQ load_state_dict: {msg}")

        # Build MotionEncoder wrapping H2VQ
        d_model = config.get("d_model", 768)
        model = cls(h2vq, d_model=d_model)
        logger.info(f"MotionEncoder built: tokens_per_t={model.tokens_per_t}, "
                     f"body={model.body_tokens_per_t}, hand={model.hand_tokens_per_t}, "
                     f"code_dim={model.code_dim}, d_model={model.d_model}")

        return model
