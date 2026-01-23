# SVG Fusion - Complete Implementation

텍스트 프롬프트에서 SVG 그래픽을 생성하는 완전한 구현체입니다.

## 🎯 주요 기능

- **VP-VAE**: SVG 벡터 표현과 픽셀 임베딩을 융합하는 Vector-Pixel VAE
- **VS-DiT**: 텍스트 조건부 잠재 확산 모델 (Vector Space Diffusion Transformer)
- **완전한 파이프라인**: SVG 파싱 → 훈련 → 생성까지 전체 과정 구현
- **실제 작동**: 더미 데이터로 즉시 테스트 가능

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 이동
cd svg_fusion

# 의존성 설치
pip install -r requirements.txt
```

### 2. 더미 데이터 생성

```bash
# 테스트용 SVG 파일 50개 생성
python create_dummy_data.py --output_dir data/svgs --num_samples 50
```

### 3. VAE 훈련

```bash
python train_vae.py \
    --svg_dir data/svgs \
    --batch_size 4 \
    --num_epochs 20 \
    --output_dir checkpoints/vae
```

### 4. DiT 훈련

```bash
python train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 4 \
    --num_epochs 30 \
    --output_dir checkpoints/dit
```

### 5. SVG 생성

```bash
python generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a red circle" \
    --num_samples 4 \
    --output_dir outputs
```

## 📐 아키텍처

### VP-VAE (Vector-Pixel VAE)

**Encoder:**
- SVG 요소 임베딩 (element_id, command_id, params)
- DINOv2 픽셀 임베딩
- Cross-attention: SVG가 픽셀 정보를 쿼리
- Transformer 레이어
- 출력: μ와 log_var

**Decoder:**
- Latent projection
- Transformer 레이어
- 출력 헤드:
  - Element 타입 (path, circle, rect, ellipse)
  - Command 타입 (M, L, C, Z 등)
  - 연속 파라미터 (좌표, 스타일)

### VS-DiT (Vector Space Diffusion Transformer)

- Timestep 임베딩
- Latent projection
- Text context projection (CLIP)
- DiT 블록:
  - AdaLN (timestep-conditioned)
  - Self-attention (latent 시퀀스)
  - Cross-attention (텍스트 컨텍스트)
  - Feed-forward
- Classifier-Free Guidance (CFG)

## 📁 프로젝트 구조

```
svg_fusion/
├── models/                     # 모델 구현
│   ├── vpvae.py               # VP-VAE
│   └── vsdit.py               # VS-DiT
├── utils/                      # 유틸리티
│   ├── svg_parser.py          # SVG 파싱 및 텐서 변환
│   ├── dataset.py             # 데이터셋 클래스
│   └── diffusion.py           # Diffusion 유틸리티
├── config.py                   # 설정
├── train_vae.py               # VAE 훈련
├── train_dit.py               # DiT 훈련
├── generate.py                # SVG 생성
├── create_dummy_data.py       # 더미 데이터 생성
└── requirements.txt           # 의존성
```

## 🔧 설정

### VAE 설정 (`config.py`)

```python
latent_dim = 128               # 잠재 벡터 차원
encoder_d_model = 512          # 인코더 모델 차원
decoder_d_model = 512          # 디코더 모델 차원
encoder_layers = 4             # 인코더 레이어 수
decoder_layers = 4             # 디코더 레이어 수
num_heads = 8                  # 어텐션 헤드 수
max_seq_len = 1024             # 최대 시퀀스 길이
```

### DiT 설정

```python
latent_dim = 128               # VAE와 동일
hidden_dim = 384               # 히든 차원
context_dim = 512              # CLIP 차원
num_blocks = 12                # DiT 블록 수
num_heads = 6                  # 어텐션 헤드 수
noise_steps = 1000             # Diffusion 스텝
```

## 📊 훈련 파라미터

### VAE 훈련

```bash
python train_vae.py \
    --svg_dir data/svgs \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --kl_warmup_steps 5000 \
    --output_dir checkpoints/vae
```

### DiT 훈련

```bash
python train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --cfg_dropout 0.25 \
    --output_dir checkpoints/dit
```

## 🎨 생성 파라미터

```bash
python generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "your text prompt" \
    --num_samples 4 \
    --cfg_scale 7.0 \
    --ddim_steps 100 \
    --output_dir outputs
```

**파라미터 설명:**
- `cfg_scale`: Classifier-Free Guidance 강도 (7-15 권장)
- `ddim_steps`: DDIM 샘플링 스텝 (50-250)
- `eta`: DDIM 확률성 (0=결정적, 1=완전 확률적)

## 🔬 기술 세부사항

### SVG 표현

SVG는 시퀀스로 표현됩니다:
- **Element ID**: path, circle, rect, ellipse
- **Command ID**: M, L, C, Z 등
- **Continuous Params**: 좌표 (8개) + 스타일 (4개)
- 모든 값은 0-255 bin으로 양자화

### DINOv2 픽셀 임베딩

- SVG를 224x224 이미지로 래스터화
- DINOv2로 임베딩 추출
- CLS 토큰을 시퀀스 길이만큼 반복

### Diffusion 프로세스

- **Forward**: 노이즈 점진적 추가
- **Reverse**: DDIM 샘플링으로 노이즈 제거
- **CFG**: 조건부/무조건부 예측 보간

## 📝 실제 데이터 사용

실제 SVG 데이터셋을 사용하려면:

1. **SVG 파일 준비**: `data/svgs/` 디렉토리에 배치
2. **캡션 생성** (선택사항): 각 SVG에 대한 텍스트 설명
3. **훈련**: VAE → DiT 순서로 훈련
4. **생성**: 훈련된 모델로 새로운 SVG 생성

## ⚡ 성능 최적화

### GPU 메모리 절약

- `--batch_size` 감소 (4 또는 2)
- `--max_seq_len` 감소 (512)
- Gradient checkpointing 활성화

### 훈련 속도 향상

- `--num_workers` 증가
- Mixed precision 훈련
- 더 작은 DINOv2 모델 사용

### 생성 품질 향상

- `--cfg_scale` 증가 (10-15)
- `--ddim_steps` 증가 (200-250)
- 더 많은 에폭 훈련

## 🐛 문제 해결

### SVG 파싱 오류
- SVG 파일이 표준 형식인지 확인
- 지원되는 요소만 사용 (path, circle, rect, ellipse)

### Out of Memory
- 배치 크기 감소
- 시퀀스 길이 감소
- Gradient accumulation 사용

### 생성 품질 낮음
- 더 많은 훈련 데이터 사용
- 더 긴 훈련
- CFG scale 조정

## 📄 라이선스

MIT License

## 🙏 감사

- DINOv2: Meta AI
- CLIP: OpenAI
- Diffusion Models: Ho et al.
