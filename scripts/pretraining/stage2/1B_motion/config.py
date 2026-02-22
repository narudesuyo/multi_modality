import os as _os
from configs.data import *
from configs.model import *

# ========================= path resolution ==========================
_HERE = _os.path.dirname(_os.path.abspath(__file__))           # .../scripts/pretraining/stage2/1B_motion
_WORK_DIR = _os.path.abspath(_os.path.join(_HERE, "../../../.."))  # multi_modality root
_EGOHAND_DIR = _os.path.abspath(_os.path.join(_WORK_DIR, "../../.."))  # 3 levels up = EgoHand
_DATA_ROOT = _os.environ.get("DATA_ROOT", "/work/narus/data")

# ========================= data ==========================
# Register video_motion dataset
available_corpus["video_motion_train"] = dict(
    anno_path=_os.path.join(_HERE, "annotation_atomic_train.json"),
    data_root=_os.path.join(_DATA_ROOT, "train/takes_clipped/egoexo"),
    motion_data_root=_os.path.join(_DATA_ROOT, "train/takes_clipped/egoexo"),
    media_type="video_motion",
    normalize_motion=False,
)

available_corpus["video_motion_assembly101_train"] = dict(
    anno_path=_os.path.join(_HERE, "annotation_assembly101_train.json"),
    data_root=_os.path.join(_DATA_ROOT, "train/takes_clipped/assembly101"),
    motion_data_root=_os.path.join(_DATA_ROOT, "train/takes_clipped/assembly101"),
    media_type="video_motion",
    normalize_motion=False,
)

available_corpus["video_motion_val"] = dict(
    anno_path=_os.path.join(_HERE, "annotation_atomic_val.json"),
    data_root=_os.path.join(_DATA_ROOT, "val/takes_clipped/egoexo"),
    motion_data_root=_os.path.join(_DATA_ROOT, "val/takes_clipped/egoexo"),
    media_type="video_motion",
    normalize_motion=False,
)

# LMDB versions (faster I/O — build with: python tools/build_lmdb.py)
available_corpus["video_motion_lmdb_train"] = dict(
    lmdb_path=_os.path.join(_DATA_ROOT, "lmdb/train.lmdb"),
    media_type="video_motion_lmdb",
)
available_corpus["video_motion_lmdb_val"] = dict(
    lmdb_path=_os.path.join(_DATA_ROOT, "lmdb/val.lmdb"),
    media_type="video_motion_lmdb",
)

# Switch between raw files and LMDB:
#   raw:  video_motion_train / video_motion_val
#   lmdb: video_motion_lmdb_train / video_motion_lmdb_val
# Assembly101 only (EgoExo4D frames not yet available)
train_file = [
    # available_corpus["video_motion_train"],              # EgoExo4D (~1048)
    available_corpus["video_motion_assembly101_train"],   # Assembly101 (~3405)
]

test_file = dict(video_motion_val=available_corpus["video_motion_val"])
test_types = ["video_motion_val"]
num_workers = 6

best_key = ["msrvtt_1k_test_match", "t2v_r1"]

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 16
batch_size_test = 1
max_txt_l = 32
motion_T = 21  # motion sequence length (matching BodyTokenize)

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
        video_reader_type="img",  # read JPG frames instead of MP4 (decord-free)
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}", video_motion="${max_txt_l}", video_motion_lmdb="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}", video_motion="${batch_size}", video_motion_lmdb="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}", video_motion="${batch_size_test}", video_motion_lmdb="${batch_size_test}"),
    motion_T="${motion_T}",
)

# ========================= model ==========================
text_enc = "bert_large"
model = dict(
    model_cls="InternVideo2_Stage2_Motion",
    vision_encoder=dict(
        # backbone
        name="pretrain_internvideo2_1b_patch14_224",
        img_size=224,
        num_frames="${num_frames}",
        tubelet_size=1,
        patch_size=14,
        d_model=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        pretrained=_os.path.join(_WORK_DIR, "scripts/pretraining/stage1/1B_ft_k710_f8.pth"),
        use_checkpoint=True,
        checkpoint_num=40,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        # clip teacher (disabled for motion experiment)
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        # mask
        video_mask_type="random",
        video_mask_ratio=0.8,
        image_mask_type="random",
        image_mask_ratio=0.5,
        sep_image_video_pos_embed=True,
        keep_temporal=False,
        only_mask=True,
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    motion_encoder=dict(
        d_model=768,
        ckpt_path=_os.path.join(_WORK_DIR, "../../../BodyTokenize/ckpt_vq/ckpt_best.pt"),
        vqvae_config=_os.path.join(_EGOHAND_DIR, "BodyTokenize/ckpt_vq/config.yaml"),
        freeze=True,
    ),
    multimodal=dict(enable=True),
    contra_dim=768,
    vm_concat_dim=768,
    temp=0.07,
    freeze_motion=True,
    find_unused_parameters=True,
)

criterion = dict(
    loss_weight=dict(
        # video-text
        vtc=1.0,
        vtm=1.0,
        mlm=1.0,
        uta=0.0,
        # motion-text
        mtc=1.0,
        mtm=1.0,
        mmlm=0.0,
        # video-motion cross
        vmc=0.0,
        # video-motion-text fused
        vmtc=1.0,
        vmtm=1.0,
        vmmlm=1.0,
    ),
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    distill_final_features=True,
    clip_loss_ratio=[1., 1.],
    uta_image_only=False,
)

optimizer = dict(
    opt="adamW",
    lr=5e-5,
    opt_betas=[0.9, 0.98],
    weight_decay=0.05,
    max_grad_norm=3.,
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=10, min_lr_multi=0.01, warmup_epochs=1)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
)

use_half_precision = True
use_bf16 = False  # V100 does not support bf16; use fp16 instead

gradient_checkpointing = True
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

# ========================= wandb ==========================
wandb = dict(
    enable=True,
    entity="narudesuyo-waseda-university",
    project="InternVideo2-Stage2-Motion",
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None
resume = False
debug = False
log_freq = 5
seed = 42

save_latest = True
auto_resume = False
jump_evaluate = False
pretrained_path = ""
save_ckpt_iter = None
delete_ds_optim_states = True

# Don't save frozen motion encoder weights (saves disk space)
no_save_params_prefix = ["motion_encoder.encB", "motion_encoder.encH"]

deepspeed = dict(
    enable=True,
    stage=2,
)
