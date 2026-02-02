# QUICKSTART

## 1) 의존성 설치

```bash
pip install -r requirements.txt
```

## 2) 데이터 준비

### 더미 데이터 생성

```bash
python scripts/create_dummy_data.py --output_dir data/svgs --num_samples 50
```

### Twemoji 데이터 다운로드 (실제 SVG)

```bash
python scripts/download_twemoji.py --output_dir data/twemoji_svg
```

Twemoji 파일명(코드포인트)을 유니코드 이름으로 변환해 간단한 캡션으로 사용합니다.

## 3) 스모크 테스트 파이프라인

### preprocess

```bash
python scripts/preprocess.py \
  --svg_dir data/twemoji_svg \
  --pattern "*.svg" \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --max_files 300
```

### train

```bash
python scripts/train.py \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --steps 300 \
  --batch_size 4 \
  --device cpu \
  --ckpt_out ./checkpoints
```

### generate

```bash
python scripts/generate.py \
  --ckpt_path ./checkpoints/model_step300.pt \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --num_eval 10 \
  --num_samples 10 \
  --out_dir ./outputs
```

## 4) DiT 샘플링 (텍스트 조건)

```bash
python scripts/generate_dit.py \
  --vae_ckpt /mnt/8TB_1/jiwoo/checkpoints/model_step300.pt \
  --dit_ckpt /mnt/8TB_1/jiwoo/checkpoints/dit/vsdit_final.pt \
  --prompt "a red circle" \
  --num_samples 4 \
  --out_dir /mnt/8TB_1/jiwoo/outputs
```
