# SVG Fusion

VP-VAE(벡터-픽셀 VAE)와 VS-DiT(디퓨전 트랜스포머)를 이용한 벡터 그래픽 생성 프로젝트입니다.

## 구성 요약

- VP-VAE: SVG 토큰 + DINOv2 픽셀 임베딩을 잠재 시퀀스로 인코딩하고 연속형 SVG 피처를 복원합니다.
- VS-DiT: 잠재 공간에서 CLIP 텍스트 조건을 사용해 epsilon 예측 디퓨전을 학습합니다.
- VAE/DiT 학습 스크립트와 데모 스크립트가 포함됩니다.

## 프로젝트 구조

```
svg_fusion/
  models/
    vpvae.py              VP-VAE 인코더/디코더
    vsdit.py              VS-DiT 블록 및 모델
  utils/
    diffusion.py          디퓨전 유틸 (스케줄, noise_latent, DDIM)
    dataset.py            SVGDataset, collate_fn
    svg_parser.py         SVG 파싱 및 텐서화
    tensorsvg.py          Tensor -> SVG 변환 유틸
    hybrid_utils.py       하이브리드 디코딩 헬퍼
  scripts/
    preprocess.py         전처리 캐시 생성 (스모크 테스트용)
    train.py              스모크 테스트용 VAE 학습
    generate.py           재구성/랜덤 샘플 생성 (스모크 테스트용)
    download_twemoji.py   Twemoji SVG 데이터 다운로드
    train_vae.py          VP-VAE 학습
    train_dit.py          VS-DiT 학습
    train_demo.py         더미 데이터 데모 학습
    create_dummy_data.py  합성 SVG 데이터 생성
    prepare_latents.py    잠재값 전처리 헬퍼
    datasetpreparation_v5.py  데이터셋 준비 스크립트
  config.py               모델/학습 설정
  README.md
  QUICKSTART.md
  requirements.txt        파이썬 의존성
```

## 빠른 시작 (스모크 테스트 파이프라인)

의존성 설치:

```bash
pip install -r requirements.txt
```

더미 데이터 생성:

```bash
python scripts/create_dummy_data.py --output_dir data/svgs --num_samples 50
```

Twemoji 데이터 다운로드(실제 데이터셋):

```bash
python scripts/download_twemoji.py --output_dir data/twemoji_svg
```

전처리 (SVG -> 캐시):

```bash
python scripts/preprocess.py \
  --svg_dir data/twemoji_svg \
  --pattern "*.svg" \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --max_files 300
```

스모크 테스트용 VAE 학습:

```bash
python scripts/train.py \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --steps 2000 \
  --batch_size 4 \
  --device cpu \
  --ckpt_out ./checkpoints
```

재구성/샘플 생성:

```bash
python scripts/generate.py \
  --ckpt_path ./checkpoints/model_step2000.pt \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --num_eval 10 \
  --num_samples 10 \
  --out_dir ./outputs
```

## 기존 학습 (선택)

VP-VAE 학습:

```bash
python scripts/train_vae.py \
  --svg_dir data/svgs \
  --batch_size 4 \
  --num_epochs 20 \
  --output_dir checkpoints/vae
```

VS-DiT 학습:

```bash
python scripts/train_dit.py \
  --svg_dir data/svgs \
  --vae_checkpoint checkpoints/vae/vpvae_final.pt \
  --batch_size 4 \
  --num_epochs 30 \
  --output_dir checkpoints/dit
```

SVG 생성:

```bash
python scripts/generate.py \
  --ckpt_path checkpoints/vae/vpvae_final.pt \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --num_eval 10 \
  --num_samples 10 \
  --out_dir ./outputs
```

## Loss 규격 (현재 코드 기준)

### VP-VAE (scripts/train_vae.py)

- 디코더 출력: `predicted_features`는 `tanh`로 [-1, 1] 범위.
- 타겟: `normalize_target(...)`가 연속 피처를 [-1, 1]로 변환.
- 마스킹 MSE:
  - `effective_len = min(L_out, L_tgt)`
  - `feature_valid_mask = (~svg_mask).unsqueeze(-1)`를 `[B,L,F]`로 확장
  - `mse_recon = sum(mse(pred*mask, tgt*mask)) / (sum(mask) + 1e-9)`
- KL:
  - `logvar`는 `exp` 계산에서 `max=80.0`으로 clamp
  - position-wise KL을 `svg_mask`로 마스킹 후 valid 평균
- 최종:
  - `loss = recon_mse_loss_weight * mse_recon + kl_weight * kl_loss`
  - KL weight는 선형으로 `kl_weight_max`까지 증가

### VS-DiT (scripts/train_dit.py)

- `t ~ Uniform(0..T-1)` 샘플.
- `(z_t, noise) = DiffusionUtils.noise_latent(z0, t, diff_params)`.
- CLIP 컨텍스트는 `last_hidden_state`와 padding mask(`True = pad`) 사용.
- CFG dropout은 샘플 단위로 conditional/unconditional 컨텍스트와 마스크를 스왑.
- 예측: `predicted_noise = dit(z_t, t, context, context_mask)`.
- 손실: `MSE(predicted_noise, noise)` (reduction="mean").

## 설정

`config.py` 참고:
- `VAEConfig`: latent 크기, 인코더/디코더 깊이, 시퀀스 길이 등
- `DiTConfig`: diffusion steps, hidden size, CLIP 컨텍스트 차원, CFG dropout 등

## 스모크 테스트 성공 조건

- 전처리 캐시가 생성됨 (.pt)
- DataLoader가 1배치 이상 통과
- 학습이 300 step 이상 진행되고 loss 로그가 출력됨
- recon SVG가 5개 이상 저장됨
- (가능하면) random 샘플 SVG가 5개 이상 저장됨
- 생성된 SVG가 cairosvg로 렌더 가능한 파일

## 참고

- `scripts/train_demo.py`는 더미 데이터로 소규모 데모 학습을 수행합니다.
- `scripts/prepare_latents.py`는 디퓨전 학습용 VAE 잠재값을 미리 계산할 때 사용합니다.
- `scripts/generate.py`는 DDIM 샘플링과 CLIP 텍스트 조건을 사용합니다.
