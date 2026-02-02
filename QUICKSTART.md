# SVG Fusion - Quick Start Guide

5遺??덉뿉 SVG Fusion???ㅽ뻾?대낫?몄슂!

## ?? ?④퀎蹂?媛?대뱶

### Step 1: ?섍꼍 ?ㅼ젙 (1遺?

```bash
# ?꾨줈?앺듃 ?붾젆?좊━濡??대룞
cd svg_fusion

# ?섏〈???ㅼ튂
pip install -r requirements.txt
```

**?꾩슂???⑦궎吏:**
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- cairosvg >= 2.7.0
- Pillow, numpy, tqdm

### Step 2: ?뚯뒪???곗씠???앹꽦 (10珥?

```bash
# 50媛쒖쓽 ?붾? SVG ?뚯씪 ?앹꽦
python scripts/create_dummy_data.py --output_dir data/svgs --num_samples 50
```

**?앹꽦?섎뒗 ?뚯씪:**
- `data/svgs/circle_*.svg` - ???꾪삎
- `data/svgs/rect_*.svg` - ?ш컖??
- `data/svgs/ellipse_*.svg` - ???
- `data/svgs/path_*.svg` - 寃쎈줈
- `data/svgs/multi_*.svg` - 蹂듯빀 ?꾪삎

### Step 3: VAE ?덈젴 (10-30遺?

```bash
# 鍮좊Ⅸ ?뚯뒪??(10遺? GPU)
python scripts/train_vae.py \
    --svg_dir data/svgs \
    --batch_size 8 \
    --num_epochs 10 \
    --output_dir checkpoints/vae

# ???섏? ?덉쭏 (30遺? GPU)
python scripts/train_vae.py \
    --svg_dir data/svgs \
    --batch_size 8 \
    --num_epochs 30 \
    --kl_warmup_steps 2000 \
    --output_dir checkpoints/vae
```

**?덈젴 吏꾪뻾 ?곹솴:**
```
Epoch 1/10: loss=2.4531, recon=2.1234, kl=0.3297
Epoch 2/10: loss=2.1245, recon=1.9123, kl=0.2122
...
Saved checkpoint: checkpoints/vae/vpvae_epoch10.pt
```

### Step 4: DiT ?덈젴 (30-60遺?

```bash
# 鍮좊Ⅸ ?뚯뒪??(30遺? GPU)
python scripts/train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 8 \
    --num_epochs 20 \
    --output_dir checkpoints/dit

# ???섏? ?덉쭏 (60遺? GPU)
python scripts/train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 1e-4 \
    --output_dir checkpoints/dit
```

### Step 5: SVG ?앹꽦 (10珥?

```bash
# 湲곕낯 ?앹꽦
python scripts/generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a red circle" \
    --num_samples 4 \
    --output_dir outputs

# 怨좏뭹吏??앹꽦
python scripts/generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a blue star with five points" \
    --num_samples 8 \
    --cfg_scale 10.0 \
    --ddim_steps 200 \
    --output_dir outputs
```

**?앹꽦???뚯씪:**
```
outputs/
?쒋?? a_red_circle_sample1.svg
?쒋?? a_red_circle_sample2.svg
?쒋?? a_red_circle_sample3.svg
?붴?? a_red_circle_sample4.svg
```

## ?뱤 ?덉긽 ?뚯슂 ?쒓컙

| ?④퀎 | CPU | GPU (RTX 3090) |
|------|-----|----------------|
| ?섍꼍 ?ㅼ젙 | 1遺?| 1遺?|
| ?곗씠???앹꽦 | 10珥?| 10珥?|
| VAE ?덈젴 (10 epochs) | 2?쒓컙 | 10遺?|
| DiT ?덈젴 (20 epochs) | 4?쒓컙 | 30遺?|
| SVG ?앹꽦 | 1遺?| 10珥?|
| **珥앺빀** | ~6?쒓컙 | ~40遺?|

## ?렞 鍮좊Ⅸ ?뚯뒪??(GPU ?놁씠)

GPU媛 ?녿떎硫????묒? ?ㅼ젙?쇰줈 ?뚯뒪??

```bash
# 1. ???곸? ?곗씠??
python scripts/create_dummy_data.py --num_samples 10

# 2. ?묒? 紐⑤뜽 (config.py ?섏젙)
# encoder_d_model = 256
# decoder_d_model = 256
# encoder_layers = 2
# decoder_layers = 2

# 3. ?묒? 諛곗튂, ?곸? ?먰룺
python scripts/train_vae.py \
    --svg_dir data/svgs \
    --batch_size 2 \
    --num_epochs 5 \
    --max_seq_len 256 \
    --output_dir checkpoints/vae

python scripts/train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 2 \
    --num_epochs 10 \
    --max_seq_len 256 \
    --output_dir checkpoints/dit

# 4. ?앹꽦
python scripts/generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a red circle" \
    --num_samples 2 \
    --ddim_steps 50
```

## ?뵩 二쇱슂 ?뚮씪誘명꽣

### ?덈젴 ?뚮씪誘명꽣

- `--batch_size`: 諛곗튂 ?ш린 (GPU 硫붾え由ъ뿉 ?곕씪 議곗젙)
- `--num_epochs`: ?먰룺 ??(??留롮쓣?섎줉 醫뗭쓬)
- `--lr`: ?숈뒿瑜?(湲곕낯媛? 1e-4)
- `--max_seq_len`: 理쒕? ?쒗??湲몄씠 (硫붾え由??곹뼢)

### ?앹꽦 ?뚮씪誘명꽣

- `--cfg_scale`: CFG 媛뺣룄 (7-15)
  - ??쓬: ?ㅼ뼇???? ?덉쭏 ??
  - ?믪쓬: ?ㅼ뼇???? ?덉쭏 ??
  
- `--ddim_steps`: ?섑뵆留??ㅽ뀦 (50-250)
  - ?곸쓬: 鍮좊쫫, ?덉쭏 ??
  - 留롮쓬: ?먮┝, ?덉쭏 ??
  
- `--eta`: ?뺣쪧??(0-1)
  - 0: 寃곗젙??
  - 1: ?뺣쪧??

## ?뮕 ?좎슜????

### 1. 泥댄겕?ъ씤???ш컻

```bash
# VAE ?덈젴 ?ш컻
python scripts/train_vae.py \
    --svg_dir data/svgs \
    --resume_from checkpoints/vae/vpvae_epoch10.pt \
    --num_epochs 20
```

### 2. ?앹꽦 ?덉쭏 媛쒖꽑

```bash
# CFG ?ㅼ???議곗젙
for cfg in 5 7 10 15; do
    python scripts/generate.py \
        --vae_checkpoint checkpoints/vae/vpvae_final.pt \
        --dit_checkpoint checkpoints/dit/vsdit_final.pt \
        --prompt "test prompt" \
        --cfg_scale $cfg \
        --num_samples 1 \
        --output_dir outputs/cfg_${cfg}
done
```

### 3. 諛곗튂 ?앹꽦

```bash
# ?щ윭 ?꾨＼?꾪듃??????앹꽦
prompts=(
    "a red circle"
    "a blue square"
    "a green triangle"
    "a yellow star"
)

for prompt in "${prompts[@]}"; do
    python scripts/generate.py \
        --vae_checkpoint checkpoints/vae/vpvae_final.pt \
        --dit_checkpoint checkpoints/dit/vsdit_final.pt \
        --prompt "$prompt" \
        --num_samples 4 \
        --output_dir "outputs/$(echo $prompt | tr ' ' '_')"
done
```

## ?맀 臾몄젣 ?닿껐

### Out of Memory

```bash
# 諛곗튂 ?ш린 媛먯냼
--batch_size 2

# ?쒗??湲몄씠 媛먯냼
--max_seq_len 256

# Gradient accumulation
--accumulation_steps 4
```

### ?덈젴???먮┝

```bash
# Workers 利앷?
--num_workers 8

# ?꾨쿋??罹먯떛 ?쒖꽦??
# (dataset.py?먯꽌 cache_embeddings=True)

# Mixed precision (PyTorch 2.0+)
--fp16
```

### ?앹꽦 寃곌낵媛 ??醫뗭쓬

1. **??留롮? ?덈젴 ?곗씠??*
2. **??湲??덈젴** (50+ epochs)
3. **CFG ?ㅼ???議곗젙** (7-15)
4. **DDIM ?ㅽ뀦 利앷?** (100-250)

## ?뱴 ?ㅼ쓬 ?④퀎

1. ??鍮좊Ⅸ ?쒖옉 ?꾨즺
2. ?뱰 [README.md](README.md) ?꾩껜 臾몄꽌 ?쎄린
3. ?렓 ?ㅼ젣 SVG ?곗씠?곕줈 ?덈젴
4. ?뵩 紐⑤뜽 ?꾪궎?띿쿂 而ㅼ뒪?곕쭏?댁쭠
5. ?뱤 寃곌낵 ?쒓컖??諛??됯?

## ?럦 異뺥븯?⑸땲??

SVG Fusion???깃났?곸쑝濡??ㅽ뻾?덉뒿?덈떎! ?댁젣 ?띿뒪?몄뿉??SVG瑜??앹꽦?????덉뒿?덈떎.

??沅곴툑???먯씠 ?덉쑝硫?README.md瑜?李멸퀬?섏꽭??
