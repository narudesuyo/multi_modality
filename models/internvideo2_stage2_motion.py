"""InternVideo2 Stage 2 model with Video + Body Motion + Text modalities.

Based on internvideo2_stage2_audiovisual.py, replacing audio with body motion.
"""

import logging

import torch
from torch import nn

from .backbones.motion import MotionEncoder
from .backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224, pretrain_internvideo2_6b_patch14_224, internvl_clip_6b
from .backbones.bert.builder import build_bert
from .criterions import MLMLoss, VTC_VTM_Loss, new_UTA_Loss
from .mask import (
    TubeMaskingGenerator,
    RandomMaskingGenerator
)

logger = logging.getLogger(__name__)


class InternVideo2_Stage2_Motion(nn.Module):
    """InternVideo2 Stage 2: Video + Body Motion + Text."""

    def __init__(self, config, tokenizer, is_pretrain=True):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.is_pretrain = is_pretrain

        self.vision_width = config.model.vision_encoder.d_model
        self.vision_proj_width = config.model.vision_encoder.clip_embed_dim
        self.text_width = config.model.text_encoder.d_model
        self.motion_width = config.model.motion_encoder.d_model

        self.contra_dim = config.model.contra_dim
        self.vm_concat_dim = config.model.vm_concat_dim

        self.loss_weight = config.criterion.loss_weight

        # ---- Vision encoder ----
        self.vision_encoder = self.build_vision_encoder()
        if config.model.get("freeze_vision", False):
            self.freeze_vision()

        self.vm_concat_vision_proj = nn.Sequential(
            nn.Linear(self.vision_width, self.vm_concat_dim),
            nn.LayerNorm(self.vm_concat_dim),
        )

        # ---- Text encoder ----
        self.text_encoder = self.build_text_encoder()
        if config.model.get("freeze_text", False):
            self.freeze_text()

        # ---- Motion encoder ----
        if self.use_motion():
            self.motion_encoder = self.build_motion_encoder()
            self.motion_proj = nn.Linear(self.motion_width, self.contra_dim)
            self.vm_concat_motion_proj = nn.Sequential(
                nn.Linear(self.motion_width, self.vm_concat_dim),
                nn.LayerNorm(self.vm_concat_dim),
            )

            if self.loss_weight.vmtc != 0 or self.loss_weight.vmtm != 0:
                self.vm_fusion = nn.Linear(2 * self.contra_dim, self.contra_dim)

            if self.loss_weight.mtm != 0:
                self.mtm_head = nn.Linear(self.text_width, 2)

            if self.loss_weight.vmtm != 0:
                self.vmtm_head = nn.Linear(self.text_width, 2)

            if config.model.get("freeze_motion", True):
                self.freeze_motion()

            # ---- Motion MLM (predict masked motion codebook indices) ----
            if getattr(self.loss_weight, 'motion_mlm', 0) != 0:
                codebook_size = self.motion_encoder.codebook_B.shape[0]
                body_tokens = self.motion_encoder.body_tokens_per_t
                hand_tokens = self.motion_encoder.hand_tokens_per_t
                motion_d = self.motion_encoder.d_model  # 768

                self.motion_mask_token = nn.Parameter(torch.zeros(1, 1, motion_d))
                self.motion_pos_embed = nn.Parameter(torch.zeros(1, 64, motion_d))
                self.motion_mlm_text_proj = nn.Linear(self.text_width, motion_d)
                self.motion_mlm_transformer = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=motion_d, nhead=8, dim_feedforward=3072,
                        batch_first=True, norm_first=True),
                    num_layers=4)
                self.motion_mlm_head_B = nn.ModuleList(
                    [nn.Linear(motion_d, codebook_size) for _ in range(body_tokens)])
                self.motion_mlm_head_H = nn.ModuleList(
                    [nn.Linear(motion_d, codebook_size) for _ in range(hand_tokens)])
                self.motion_mlm_mask_ratio = config.criterion.get(
                    'motion_mlm_mask_ratio', 0.15)

                nn.init.normal_(self.motion_mask_token, std=0.02)
                nn.init.normal_(self.motion_pos_embed, std=0.02)
                logger.info(
                    f"Motion MLM enabled: codebook_size={codebook_size}, "
                    f"body_heads={body_tokens}, hand_heads={hand_tokens}, "
                    f"mask_ratio={self.motion_mlm_mask_ratio}")

        # ---- Projection layers ----
        self.vision_proj = nn.Linear(self.vision_proj_width, self.contra_dim)
        self.text_proj = nn.Linear(self.text_width, self.contra_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.itm_head = nn.Linear(self.text_width, 2)

        # ---- Criterions ----
        self.loss_weight = config.criterion.loss_weight
        self.criterion_uta = new_UTA_Loss(
            config.criterion.distill_final_features,
            config.criterion.clip_loss_ratio,
        )
        self.criterion_vtc_vtm = VTC_VTM_Loss(config.criterion.vtm_hard_neg)
        self.criterion_mlm = MLMLoss(config.criterion.mlm_masking_prob, tokenizer)
        self.uta_image_only = config.criterion.get('uta_image_only', False)
        logger.info(f"uta_image_only={self.uta_image_only}")

        # ---- Unfreeze specific keys ----
        logger.info(f"unfreeze_keys: {config.model.get('unfreeze_keys', [])}")
        for k, p in self.named_parameters():
            for uk in config.model.get("unfreeze_keys", []):
                if uk in k:
                    p.requires_grad = True
                    logger.info(f"unfreeze_key: {k}")

        self.num_test_segments = config.get("num_test_segments", 1)
        logger.info(f"num_test_segments={self.num_test_segments}")

    # ================================================================
    # Utility methods
    # ================================================================

    def use_motion(self):
        lw = self.loss_weight
        return (lw.mtc != 0 or lw.vmc != 0 or lw.vmtc != 0 or
                lw.mtm != 0 or lw.vmtm != 0 or lw.mmlm != 0 or lw.vmmlm != 0 or
                getattr(lw, 'motion_mlm', 0) != 0)

    def freeze_vision(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def freeze_text(self):
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def freeze_motion(self):
        for p in self.motion_encoder.parameters():
            p.requires_grad = False
        logger.info("Motion encoder is frozen.")

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        return ret

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    # ================================================================
    # Forward methods
    # ================================================================

    def forward(self, media, text, idx, media_type='video'):
        if media_type == 'video_motion':
            video = media[0]
            motion_indices = media[1]
            return self.forward_video_motion_indices(video, motion_indices, text, idx)
        elif media_type in ['image', 'video']:
            return self.forward_image_video(media, text, idx)
        else:
            raise NotImplementedError(f"Not supported: {media_type}")

    def forward_image_video(self, image, text, idx):
        self.clip_contrastive_temperature()
        T = image.shape[1]
        use_image = True if T == 1 else False

        vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis = self.encode_vision(image)
        text_embeds, pooled_text_embeds = self.encode_text(text)

        vision_proj = self.vision_proj(pooled_vision_embeds)
        text_proj = self.text_proj(pooled_text_embeds)

        # UTA loss
        if self.loss_weight.uta != 0:
            if self.uta_image_only and not use_image:
                loss_uta = torch.tensor(0)
            else:
                loss_uta = self.criterion_uta.uta_loss(student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis)
        else:
            loss_uta = torch.tensor(0)

        # VTC loss
        if self.loss_weight.vtc != 0:
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj, text_proj, idx, self.temp, all_gather=True)
        else:
            loss_vtc = torch.tensor(0)

        # VTM loss
        if self.loss_weight.vtm != 0:
            loss_vtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(), self.itm_head, self.temp,
                vision_embeds, text_embeds, vision_proj, text_proj,
                text.attention_mask, idx)
        else:
            loss_vtm = torch.tensor(0)

        # MLM loss
        if self.is_pretrain and self.loss_weight.mlm != 0:
            loss_mlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, vision_embeds, None)
        else:
            loss_mlm = torch.tensor(0)

        return dict(
            loss_uta=loss_uta * self.loss_weight.uta,
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_mlm=loss_mlm * self.loss_weight.mlm,
        )

    def forward_video_motion_indices(self, video, motion_indices, text, idx):
        """Forward for video_motion media type: video + tok_pose indices + text."""
        self.clip_contrastive_temperature()

        text_embeds, pooled_text_embeds = self.encode_text(text)
        text_proj = self.text_proj(pooled_text_embeds)

        # Vision
        vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis = self.encode_vision(video)
        vision_proj = self.vision_proj(pooled_vision_embeds).squeeze(dim=1)

        # Motion (from token indices → codebook lookup)
        if self.use_motion():
            motion_embeds, pooled_motion_embeds = self.encode_motion_from_indices(motion_indices)
            motion_proj = self.motion_proj(pooled_motion_embeds)

        # ---- Losses ----

        # UTA loss
        if self.loss_weight.uta != 0:
            if self.uta_image_only:
                loss_uta = torch.tensor(0)
            else:
                loss_uta = self.criterion_uta.uta_loss(student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis)
        else:
            loss_uta = torch.tensor(0)

        # VTC loss (video-text contrastive)
        if self.loss_weight.vtc != 0:
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj, text_proj, idx, self.temp, all_gather=True)
        else:
            loss_vtc = torch.tensor(0)

        # MTC loss (motion-text contrastive)
        if self.loss_weight.mtc != 0:
            loss_mtc = self.criterion_vtc_vtm.vtc_loss(
                motion_proj, text_proj, idx, self.temp, all_gather=True)
        else:
            loss_mtc = torch.tensor(0)

        # VMC loss (video-motion contrastive)
        if self.loss_weight.vmc != 0:
            loss_vmc = self.criterion_vtc_vtm.vtc_loss(
                motion_proj, vision_proj, idx, self.temp, all_gather=True)
        else:
            loss_vmc = torch.tensor(0)

        # VMTC loss (video-motion fused - text contrastive)
        if self.loss_weight.vmtc != 0:
            loss_vmtc = self.criterion_vtc_vtm.vtc_loss(
                self.vm_fusion(torch.cat([motion_proj, vision_proj], dim=-1)),
                text_proj, idx, self.temp, all_gather=True)
        else:
            loss_vmtc = torch.tensor(0)

        # VTM loss (video-text matching)
        if self.loss_weight.vtm != 0:
            loss_vtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(), self.itm_head, self.temp,
                vision_embeds, text_embeds, vision_proj, text_proj,
                text.attention_mask, idx)
        else:
            loss_vtm = torch.tensor(0)

        # MTM loss (motion-text matching)
        if self.loss_weight.mtm != 0:
            loss_mtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(), self.mtm_head, self.temp,
                motion_embeds, text_embeds, motion_proj, text_proj,
                text.attention_mask, idx)
        else:
            loss_mtm = torch.tensor(0)

        # VMTM loss (video-motion fused - text matching)
        if self.loss_weight.vmtm != 0:
            loss_vmtm = self.criterion_vtc_vtm.vtm_loss(
                self.get_text_encoder(), self.vmtm_head, self.temp,
                torch.cat([motion_embeds, vision_embeds], dim=-2),
                text_embeds,
                self.vm_fusion(torch.cat([motion_proj, vision_proj], dim=-1)),
                text_proj, text.attention_mask, idx)
        else:
            loss_vmtm = torch.tensor(0)

        # MLM loss (vision context)
        if self.is_pretrain and self.loss_weight.mlm != 0:
            loss_mlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, vision_embeds, None)
        else:
            loss_mlm = torch.tensor(0)

        # MMLM loss (motion context)
        if self.is_pretrain and self.loss_weight.mmlm != 0:
            loss_mmlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text, motion_embeds, None)
        else:
            loss_mmlm = torch.tensor(0)

        # VMMLM loss (video-motion fused context)
        if self.is_pretrain and self.loss_weight.vmmlm != 0:
            loss_vmmlm = self.criterion_mlm.mlm_loss(
                self.text_encoder, text,
                torch.cat([motion_embeds, vision_embeds], dim=-2), None)
        else:
            loss_vmmlm = torch.tensor(0)

        # Motion MLM loss (video+text+motion context → predict masked motion tokens)
        if self.is_pretrain and getattr(self.loss_weight, 'motion_mlm', 0) != 0:
            loss_motion_mlm = self.forward_motion_mlm(
                motion_indices, text_embeds, text.attention_mask, vision_embeds)
        else:
            loss_motion_mlm = torch.tensor(0)

        return dict(
            loss_uta=loss_uta * self.loss_weight.uta,
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_mlm=loss_mlm * self.loss_weight.mlm,
            # motion-related
            loss_vmc=loss_vmc * self.loss_weight.vmc,
            loss_mtc=loss_mtc * self.loss_weight.mtc,
            loss_vmtc=loss_vmtc * self.loss_weight.vmtc,
            loss_mtm=loss_mtm * self.loss_weight.mtm,
            loss_vmtm=loss_vmtm * self.loss_weight.vmtm,
            loss_mmlm=loss_mmlm * self.loss_weight.mmlm,
            loss_vmmlm=loss_vmmlm * self.loss_weight.vmmlm,
            loss_motion_mlm=loss_motion_mlm * getattr(self.loss_weight, 'motion_mlm', 0),
        )

    # ================================================================
    # Encoder builders and helpers
    # ================================================================

    def encode_teacher(self, image):
        B, C, T, H, W = image.shape
        mask_type = self.image_mask_type if T == 1 else self.video_mask_type
        window_size = self.image_window_size if T == 1 else self.video_window_size
        mask_ratio = self.image_mask_ratio if T == 1 else self.video_mask_ratio

        if (self.uta_image_only and T != 1) or self.config.model.vision_encoder.get('only_mask', False):
            if mask_type == 'tube':
                mask = TubeMaskingGenerator(window_size, mask_ratio, B)
            elif mask_type == 'random':
                mask = RandomMaskingGenerator(window_size, mask_ratio, B)
            elif mask_type == 'none':
                return None, None, None
            else:
                raise NotImplementedError

            mask = mask.view(B, -1).to(torch.bool)
            mask = torch.cat((torch.zeros(B, 1).to(mask.device), mask), dim=1)
            mask = mask.to(torch.bool)
            return mask, None, None

        if self.clip_teacher is None or self.loss_weight.uta == 0:
            return None, None, None

        if H != self.clip_img_size:
            image = torch.nn.functional.interpolate(
                image.reshape(B, C * T, H, W),
                size=(self.clip_img_size, self.clip_img_size),
                mode='bicubic', align_corners=False
            )
            image = image.view(B, C, T, self.clip_img_size, self.clip_img_size)

        with torch.no_grad():
            if mask_type == 'tube':
                mask = TubeMaskingGenerator(window_size, mask_ratio, B)
                norm_clip_middle, norm_clip_final, attn = self.clip_teacher(image)
            elif mask_type == 'random':
                mask = RandomMaskingGenerator(window_size, mask_ratio, B)
                norm_clip_middle, norm_clip_final, attn = self.clip_teacher(image)
            elif mask_type == 'attention':
                norm_clip_middle, norm_clip_final, attn = self.clip_teacher(image)
                BT, N = attn.shape
                N_vis = N - int(N * mask_ratio)
                importance = torch.multinomial(attn, N)
                mask = torch.ones((BT, N))
                pos1 = torch.arange(BT).view(-1, 1).repeat(1, N_vis)
                pos2 = importance[:, :N_vis]
                mask[pos1, pos2] = 0
            else:
                raise NotImplementedError

            mask = mask.view(B, -1).to(torch.bool)
            mask = torch.cat((torch.zeros(B, 1), mask), dim=1)
            mask = mask.to(torch.bool)

            C_CLIP = norm_clip_middle.shape[-1]
            if len(norm_clip_middle.shape) == 4:
                K = norm_clip_middle.shape[0]
                clip_mask = mask.unsqueeze(0).repeat(K, 1, 1)
                targets_clip_middle_vis = norm_clip_middle[~clip_mask].reshape(K, B, -1, C_CLIP)
            else:
                clip_mask = mask
                targets_clip_middle_vis = norm_clip_middle[~clip_mask].reshape(B, -1, C_CLIP)

            targets_clip_final_vis = norm_clip_final

        return mask, targets_clip_middle_vis, targets_clip_final_vis

    def encode_vision(self, image, test=False):
        B, T, C, H, W = image.shape
        use_image = True if T == 1 else False
        if not use_image and test and self.num_test_segments != 1:
            assert T // self.num_test_segments == self.config.model.vision_encoder.num_frames
            image = image.view(B, T // self.num_test_segments, self.num_test_segments, C, H, W).permute(0, 2, 1, 3, 4, 5).reshape(B * self.num_test_segments, T // self.num_test_segments, C, H, W)

        image = image.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]

        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(image, None, use_image)
            vision_embeds = self.vm_concat_vision_proj(vision_embeds)

            if not use_image and self.num_test_segments != 1:
                n, d = vision_embeds.shape[-2], vision_embeds.shape[-1]
                vision_embeds = vision_embeds.reshape(B, self.num_test_segments, n, d).mean(1, keepdim=False)
                d = pooled_vision_embeds.shape[-1]
                pooled_vision_embeds = pooled_vision_embeds.reshape(B, self.num_test_segments, d).mean(1, keepdim=False)

            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(image)
            vision_embeds, pooled_vision_embeds, student_output, student_output_final = self.vision_encoder(image, mask, use_image)
            vision_embeds = self.vm_concat_vision_proj(vision_embeds)
            return vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis

    def encode_motion(self, body_motion, hand_motion, test=False):
        motion_embeds, pooled_motion_embeds = self.motion_encoder(body_motion, hand_motion)
        motion_embeds = self.vm_concat_motion_proj(motion_embeds)
        return motion_embeds, pooled_motion_embeds

    def encode_motion_from_indices(self, motion_indices, test=False):
        """Encode motion from VQ-VAE token indices via codebook lookup."""
        motion_embeds, pooled_motion_embeds = self.motion_encoder.forward_from_indices(motion_indices)
        motion_embeds = self.vm_concat_motion_proj(motion_embeds)
        return motion_embeds, pooled_motion_embeds

    def forward_motion_mlm(self, motion_indices, text_embeds, text_attention_mask,
                           vision_embeds=None):
        """Predict masked motion timestep codebook indices (Motion MLM).

        Context: video + text + unmasked motion → predict masked motion tokens.

        Args:
            motion_indices: [B, N] flat token indices
            text_embeds: [B, seq_len, text_width] from text encoder
            text_attention_mask: [B, seq_len] attention mask (1=real, 0=pad)
            vision_embeds: [B, n_vis, 768] from vision encoder (optional)
        Returns:
            loss: scalar cross-entropy loss averaged over 8 prediction heads
        """
        B = motion_indices.shape[0]
        tokens_per_t = self.motion_encoder.tokens_per_t
        body_tokens = self.motion_encoder.body_tokens_per_t
        hand_tokens = self.motion_encoder.hand_tokens_per_t

        # Reshape to [B, T, 8] for target indices
        indices = motion_indices.reshape(B, -1, tokens_per_t)  # [B, T, 8]
        T = indices.shape[1]

        # Get motion embeddings from frozen encoder (stop grad)
        with torch.no_grad():
            motion_embeds, _ = self.motion_encoder.forward_from_indices(motion_indices)
        motion_embeds = motion_embeds.detach()  # [B, T, 768]

        # Random mask timesteps
        num_mask = max(1, int(T * self.motion_mlm_mask_ratio))
        noise = torch.rand(B, T, device=indices.device)
        mask_topk = noise.argsort(dim=1)[:, :num_mask]  # [B, num_mask]
        mask = torch.zeros(B, T, dtype=torch.bool, device=indices.device)
        mask.scatter_(1, mask_topk, True)

        # Replace masked positions with learnable [MASK] token
        mask_3d = mask.unsqueeze(-1)  # [B, T, 1]
        masked_embeds = torch.where(
            mask_3d, self.motion_mask_token.expand(B, T, -1), motion_embeds)

        # Add positional embedding
        masked_embeds = masked_embeds + self.motion_pos_embed[:, :T, :]

        # Build memory: text (+ vision if available)
        text_memory = self.motion_mlm_text_proj(text_embeds)  # [B, seq_len, 768]

        if vision_embeds is not None:
            # vision_embeds already [B, n_vis, 768] via vm_concat_vision_proj
            memory = torch.cat([text_memory, vision_embeds], dim=1)
            vis_mask = torch.ones(B, vision_embeds.shape[1],
                                  device=text_attention_mask.device,
                                  dtype=text_attention_mask.dtype)
            full_mask = torch.cat([text_attention_mask, vis_mask], dim=1)
            memory_key_padding_mask = ~full_mask.bool()
        else:
            memory = text_memory
            memory_key_padding_mask = (
                ~text_attention_mask.bool() if text_attention_mask is not None
                else None)

        # TransformerDecoder: self-attn on motion + cross-attn to text+vision
        decoded = self.motion_mlm_transformer(
            tgt=masked_embeds,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, T, 768]

        # Extract masked positions and predict codebook indices
        masked_decoded = decoded[mask]   # [num_masked_total, 768]
        masked_targets = indices[mask]   # [num_masked_total, 8]

        targets_B = masked_targets[:, :body_tokens]  # [num_masked_total, 4]
        targets_H = masked_targets[:, body_tokens:]   # [num_masked_total, 4]

        # Predict with 8 heads (4 body + 4 hand), average loss
        loss = torch.tensor(0.0, device=masked_decoded.device, dtype=masked_decoded.dtype)
        for i in range(body_tokens):
            logits = self.motion_mlm_head_B[i](masked_decoded)
            loss = loss + nn.functional.cross_entropy(logits.float(), targets_B[:, i])
        for i in range(hand_tokens):
            logits = self.motion_mlm_head_H[i](masked_decoded)
            loss = loss + nn.functional.cross_entropy(logits.float(), targets_H[:, i])

        loss = loss / (body_tokens + hand_tokens)
        return loss

    def encode_text(self, text):
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        self.temp.clamp_(min_val, max_val)

    def build_vision_encoder(self):
        encoder_name = self.config.model.vision_encoder.name
        logger.info(f"Build vision_encoder: {encoder_name}")
        if encoder_name == 'pretrain_internvideo2_1b_patch14_224':
            vision_encoder = pretrain_internvideo2_1b_patch14_224(self.config.model)
        elif encoder_name == 'pretrain_internvideo2_6b_patch14_224':
            vision_encoder = pretrain_internvideo2_6b_patch14_224(self.config.model)
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        # CLIP teacher (for UTA distillation)
        teacher_name = self.config.model.vision_encoder.clip_teacher
        self.clip_teacher = None
        if teacher_name is not None and teacher_name != "none":
            assert teacher_name == 'internvl_clip_6b'
            self.clip_teacher = internvl_clip_6b(
                img_size=self.config.model.vision_encoder.clip_input_resolution,
                clip_norm_type=self.config.model.vision_encoder.clip_norm_type,
                return_attn=True,
                clip_return_layer=self.config.model.vision_encoder.clip_return_layer,
                clip_return_interval=self.config.model.vision_encoder.clip_teacher_return_interval
            )
            for p in self.clip_teacher.parameters():
                p.requires_grad = False

        # Mask parameters
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_input_resolution
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (num_frames // tublet_size, img_size // patch_size, img_size // patch_size)
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio

        return vision_encoder

    def build_text_encoder(self):
        encoder_name = self.config.model.text_encoder.name
        logger.info(f"Build text_encoder {encoder_name}")
        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
                encoder_width=self.vm_concat_dim
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")
        return text_encoder

    def build_motion_encoder(self):
        logger.info("Build motion_encoder (BodyTokenize H2VQ)")
        motion_encoder = MotionEncoder.from_pretrained(self.config.model.motion_encoder)
        return motion_encoder

    def get_text_encoder(self):
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder
