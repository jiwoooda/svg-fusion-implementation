"""
SVG Fusion 설정 파일
모든 하이퍼파라미터와 경로를 중앙에서 관리
"""

# ============================================================================
# 경로 설정
# ============================================================================
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"

# ============================================================================
# VAE 설정
# ============================================================================
VAE_CONFIG = {
    # 모델 구조
    "latent_dim": 128,
    "encoder_d_model": 512,
    "decoder_d_model": 512,
    "encoder_layers": 4,
    "decoder_layers": 4,
    "num_heads": 8,
    
    # SVG 파라미터
    "max_seq_len": 1024,
    "num_element_types": 7,
    "num_command_types": 14,
    "num_continuous_params": 12,
    "element_embed_dim": 64,
    "command_embed_dim": 64,
    
    # 이미지 파라미터
    "pixel_feature_dim": 384,  # DINOv2-small
    
    # 훈련
    "learning_rate": 3e-4,
    "batch_size": 16,
    "total_steps": 15000,
    "warmup_steps": 300,
    "kl_weight_max": 0.5,
}

# ============================================================================
# DiT 설정
# ============================================================================
DIT_CONFIG = {
    # 모델 구조
    "latent_dim": 128,  # VAE latent_dim과 동일
    "hidden_dim": 384,
    "context_dim": 768,  # CLIP 차원
    "num_blocks": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "dropout": 0.1,
    
    # 확산
    "noise_steps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    
    # 훈련
    "learning_rate": 1e-4,
    "batch_size": 8,
    "total_steps": 50000,
    "warmup_steps": 500,
    "cfg_dropout_prob": 0.25,
}

# ============================================================================
# 샘플링 설정
# ============================================================================
SAMPLING_CONFIG = {
    "ddim_steps": 100,
    "cfg_scale": 7.0,
    "eta": 0.0,
    "num_samples": 4,
}

# ============================================================================
# CLIP 모델 경로
# ============================================================================
CLIP_MODEL_PATH = "openai/clip-vit-large-patch14"
