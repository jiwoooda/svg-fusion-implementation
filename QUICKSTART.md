# SVG Fusion - Quick Start Guide

5분 안에 SVG Fusion을 실행해보세요!

## 🚀 단계별 가이드

### Step 1: 환경 설정 (1분)

```bash
# 프로젝트 디렉토리로 이동
cd svg_fusion

# 의존성 설치
pip install -r requirements.txt
```

**필요한 패키지:**
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- cairosvg >= 2.7.0
- Pillow, numpy, tqdm

### Step 2: 테스트 데이터 생성 (10초)

```bash
# 50개의 더미 SVG 파일 생성
python create_dummy_data.py --output_dir data/svgs --num_samples 50
```

**생성되는 파일:**
- `data/svgs/circle_*.svg` - 원 도형
- `data/svgs/rect_*.svg` - 사각형
- `data/svgs/ellipse_*.svg` - 타원
- `data/svgs/path_*.svg` - 경로
- `data/svgs/multi_*.svg` - 복합 도형

### Step 3: VAE 훈련 (10-30분)

```bash
# 빠른 테스트 (10분, GPU)
python train_vae.py \
    --svg_dir data/svgs \
    --batch_size 8 \
    --num_epochs 10 \
    --output_dir checkpoints/vae

# 더 나은 품질 (30분, GPU)
python train_vae.py \
    --svg_dir data/svgs \
    --batch_size 8 \
    --num_epochs 30 \
    --kl_warmup_steps 2000 \
    --output_dir checkpoints/vae
```

**훈련 진행 상황:**
```
Epoch 1/10: loss=2.4531, recon=2.1234, kl=0.3297
Epoch 2/10: loss=2.1245, recon=1.9123, kl=0.2122
...
Saved checkpoint: checkpoints/vae/vpvae_epoch10.pt
```

### Step 4: DiT 훈련 (30-60분)

```bash
# 빠른 테스트 (30분, GPU)
python train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 8 \
    --num_epochs 20 \
    --output_dir checkpoints/dit

# 더 나은 품질 (60분, GPU)
python train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 1e-4 \
    --output_dir checkpoints/dit
```

### Step 5: SVG 생성 (10초)

```bash
# 기본 생성
python generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a red circle" \
    --num_samples 4 \
    --output_dir outputs

# 고품질 생성
python generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a blue star with five points" \
    --num_samples 8 \
    --cfg_scale 10.0 \
    --ddim_steps 200 \
    --output_dir outputs
```

**생성된 파일:**
```
outputs/
├── a_red_circle_sample1.svg
├── a_red_circle_sample2.svg
├── a_red_circle_sample3.svg
└── a_red_circle_sample4.svg
```

## 📊 예상 소요 시간

| 단계 | CPU | GPU (RTX 3090) |
|------|-----|----------------|
| 환경 설정 | 1분 | 1분 |
| 데이터 생성 | 10초 | 10초 |
| VAE 훈련 (10 epochs) | 2시간 | 10분 |
| DiT 훈련 (20 epochs) | 4시간 | 30분 |
| SVG 생성 | 1분 | 10초 |
| **총합** | ~6시간 | ~40분 |

## 🎯 빠른 테스트 (GPU 없이)

GPU가 없다면 더 작은 설정으로 테스트:

```bash
# 1. 더 적은 데이터
python create_dummy_data.py --num_samples 10

# 2. 작은 모델 (config.py 수정)
# encoder_d_model = 256
# decoder_d_model = 256
# encoder_layers = 2
# decoder_layers = 2

# 3. 작은 배치, 적은 에폭
python train_vae.py \
    --svg_dir data/svgs \
    --batch_size 2 \
    --num_epochs 5 \
    --max_seq_len 256 \
    --output_dir checkpoints/vae

python train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 2 \
    --num_epochs 10 \
    --max_seq_len 256 \
    --output_dir checkpoints/dit

# 4. 생성
python generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a red circle" \
    --num_samples 2 \
    --ddim_steps 50
```

## 🔧 주요 파라미터

### 훈련 파라미터

- `--batch_size`: 배치 크기 (GPU 메모리에 따라 조정)
- `--num_epochs`: 에폭 수 (더 많을수록 좋음)
- `--lr`: 학습률 (기본값: 1e-4)
- `--max_seq_len`: 최대 시퀀스 길이 (메모리 영향)

### 생성 파라미터

- `--cfg_scale`: CFG 강도 (7-15)
  - 낮음: 다양성 ↑, 품질 ↓
  - 높음: 다양성 ↓, 품질 ↑
  
- `--ddim_steps`: 샘플링 스텝 (50-250)
  - 적음: 빠름, 품질 ↓
  - 많음: 느림, 품질 ↑
  
- `--eta`: 확률성 (0-1)
  - 0: 결정적
  - 1: 확률적

## 💡 유용한 팁

### 1. 체크포인트 재개

```bash
# VAE 훈련 재개
python train_vae.py \
    --svg_dir data/svgs \
    --resume_from checkpoints/vae/vpvae_epoch10.pt \
    --num_epochs 20
```

### 2. 생성 품질 개선

```bash
# CFG 스케일 조정
for cfg in 5 7 10 15; do
    python generate.py \
        --vae_checkpoint checkpoints/vae/vpvae_final.pt \
        --dit_checkpoint checkpoints/dit/vsdit_final.pt \
        --prompt "test prompt" \
        --cfg_scale $cfg \
        --num_samples 1 \
        --output_dir outputs/cfg_${cfg}
done
```

### 3. 배치 생성

```bash
# 여러 프롬프트에 대해 생성
prompts=(
    "a red circle"
    "a blue square"
    "a green triangle"
    "a yellow star"
)

for prompt in "${prompts[@]}"; do
    python generate.py \
        --vae_checkpoint checkpoints/vae/vpvae_final.pt \
        --dit_checkpoint checkpoints/dit/vsdit_final.pt \
        --prompt "$prompt" \
        --num_samples 4 \
        --output_dir "outputs/$(echo $prompt | tr ' ' '_')"
done
```

## 🐛 문제 해결

### Out of Memory

```bash
# 배치 크기 감소
--batch_size 2

# 시퀀스 길이 감소
--max_seq_len 256

# Gradient accumulation
--accumulation_steps 4
```

### 훈련이 느림

```bash
# Workers 증가
--num_workers 8

# 임베딩 캐싱 활성화
# (dataset.py에서 cache_embeddings=True)

# Mixed precision (PyTorch 2.0+)
--fp16
```

### 생성 결과가 안 좋음

1. **더 많은 훈련 데이터**
2. **더 긴 훈련** (50+ epochs)
3. **CFG 스케일 조정** (7-15)
4. **DDIM 스텝 증가** (100-250)

## 📚 다음 단계

1. ✅ 빠른 시작 완료
2. 📖 [README.md](README.md) 전체 문서 읽기
3. 🎨 실제 SVG 데이터로 훈련
4. 🔧 모델 아키텍처 커스터마이징
5. 📊 결과 시각화 및 평가

## 🎉 축하합니다!

SVG Fusion을 성공적으로 실행했습니다! 이제 텍스트에서 SVG를 생성할 수 있습니다.

더 궁금한 점이 있으면 README.md를 참고하세요.
