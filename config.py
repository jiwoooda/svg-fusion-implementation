"""
SVG Fusion 설정 파일
모든 하이퍼파라미터와 경로를 중앙에서 관리
"""

from dataclasses import dataclass, field


# ============================================================================
# 경로 설정
# ============================================================================
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"


# ============================================================================
# Vocab 정의 (스펙 기준)
# ============================================================================
ELEMENT_TYPES = {
    '<BOS>': 0, 'rect': 1, 'circle': 2, 'ellipse': 3,
    'path': 4, '<EOS>': 5, '<PAD>': 6
}

PATH_COMMAND_TYPES = {
    'NO_CMD': 0, 'm': 1, 'l': 2, 'h': 3, 'v': 4,
    'c': 5, 's': 6, 'q': 7, 't': 8, 'a': 9,
    'z': 10, 'STYLE': 11, 'DEF': 12
}

NUM_ELEMENT_TYPES = len(ELEMENT_TYPES)   # 7
NUM_COMMAND_TYPES = len(PATH_COMMAND_TYPES)  # 13
NUM_CONTINUOUS_PARAMS = 12  # 8 geom + 4 style
N_BINS = 256
DEFAULT_PARAM_VAL = 0

BOS_IDX = ELEMENT_TYPES['<BOS>']  # 0
EOS_IDX = ELEMENT_TYPES['<EOS>']  # 5
PAD_IDX = ELEMENT_TYPES['<PAD>']  # 6


# ============================================================================
# VAE 설정
# ============================================================================
@dataclass
class VAEConfig:
    # 모델 구조
    latent_dim: int = 128
    encoder_d_model: int = 512
    decoder_d_model: int = 512
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8

    # SVG 파라미터
    max_seq_len: int = 1024
    num_element_types: int = NUM_ELEMENT_TYPES  # 7
    num_command_types: int = NUM_COMMAND_TYPES   # 13
    num_continuous_params: int = NUM_CONTINUOUS_PARAMS  # 12
    element_embed_dim: int = 64
    command_embed_dim: int = 64

    # 이미지 파라미터
    pixel_embed_dim: int = 384  # DINOv2-small

    # 훈련
    learning_rate: float = 3e-4
    batch_size: int = 16
    total_steps: int = 15000
    warmup_steps: int = 300
    kl_weight_max: float = 0.5


# ============================================================================
# DiT 설정
# ============================================================================
@dataclass
class DiTConfig:
    # 모델 구조
    latent_dim: int = 128  # VAE latent_dim과 동일
    hidden_dim: int = 384
    context_dim: int = 768  # CLIP 차원
    num_blocks: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # 확산
    noise_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # 훈련
    learning_rate: float = 1e-4
    batch_size: int = 8
    total_steps: int = 50000
    warmup_steps: int = 500
    cfg_dropout_prob: float = 0.10  # 논문 기준 10%


# ============================================================================
# 샘플링 설정
# ============================================================================
@dataclass
class SamplingConfig:
    ddim_steps: int = 100
    cfg_scale: float = 7.0
    eta: float = 0.0
    num_samples: int = 4


# ============================================================================
# CLIP 모델 경로
# ============================================================================
CLIP_MODEL_PATH = "openai/clip-vit-large-patch14"
