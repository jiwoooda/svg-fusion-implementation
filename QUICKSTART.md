# SVG Fusion - Quick Start Guide

5ë¶??ˆì— SVG Fusion???¤í–‰?´ë³´?¸ìš”!

## ?? ?¨ê³„ë³?ê°€?´ë“œ

### Step 1: ?˜ê²½ ?¤ì • (1ë¶?

```bash
# ?„ë¡œ?íŠ¸ ?”ë ‰? ë¦¬ë¡??´ë™
cd svg_fusion

# ?˜ì¡´???¤ì¹˜
pip install -r requirements.txt
```

**?„ìš”???¨í‚¤ì§€:**
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- cairosvg >= 2.7.0
- Pillow, numpy, tqdm

### Step 2: ?ŒìŠ¤???°ì´???ì„± (10ì´?

```bash
# 50ê°œì˜ ?”ë? SVG ?Œì¼ ?ì„±
python scripts/create_dummy_data.py --output_dir data/svgs --num_samples 50
```

**?ì„±?˜ëŠ” ?Œì¼:**
- `data/svgs/circle_*.svg` - ???„í˜•
- `data/svgs/rect_*.svg` - ?¬ê°??
- `data/svgs/ellipse_*.svg` - ?€??
- `data/svgs/path_*.svg` - ê²½ë¡œ
- `data/svgs/multi_*.svg` - ë³µí•© ?„í˜•

### Step 3: VAE ?ˆë ¨ (10-30ë¶?

```bash
# ë¹ ë¥¸ ?ŒìŠ¤??(10ë¶? GPU)
python scripts/train_vae.py \
    --svg_dir data/svgs \
    --batch_size 8 \
    --num_epochs 10 \
    --output_dir checkpoints/vae

# ???˜ì? ?ˆì§ˆ (30ë¶? GPU)
python scripts/train_vae.py \
    --svg_dir data/svgs \
    --batch_size 8 \
    --num_epochs 30 \
    --kl_warmup_steps 2000 \
    --output_dir checkpoints/vae
```

**?ˆë ¨ ì§„í–‰ ?í™©:**
```
Epoch 1/10: loss=2.4531, recon=2.1234, kl=0.3297
Epoch 2/10: loss=2.1245, recon=1.9123, kl=0.2122
...
Saved checkpoint: checkpoints/vae/vpvae_epoch10.pt
```

### Step 4: DiT ?ˆë ¨ (30-60ë¶?

```bash
# ë¹ ë¥¸ ?ŒìŠ¤??(30ë¶? GPU)
python scripts/train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 8 \
    --num_epochs 20 \
    --output_dir checkpoints/dit

# ???˜ì? ?ˆì§ˆ (60ë¶? GPU)
python scripts/train_dit.py \
    --svg_dir data/svgs \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 1e-4 \
    --output_dir checkpoints/dit
```

### Step 5: SVG ?ì„± (10ì´?

```bash
# ê¸°ë³¸ ?ì„±
python scripts/generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a red circle" \
    --num_samples 4 \
    --output_dir outputs

# ê³ í’ˆì§??ì„±
python scripts/generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a blue star with five points" \
    --num_samples 8 \
    --cfg_scale 10.0 \
    --ddim_steps 200 \
    --output_dir outputs
```

**?ì„±???Œì¼:**
```
outputs/
?œâ??€ a_red_circle_sample1.svg
?œâ??€ a_red_circle_sample2.svg
?œâ??€ a_red_circle_sample3.svg
?”â??€ a_red_circle_sample4.svg
```

## ?“Š ?ˆìƒ ?Œìš” ?œê°„

| ?¨ê³„ | CPU | GPU (RTX 3090) |
|------|-----|----------------|
| ?˜ê²½ ?¤ì • | 1ë¶?| 1ë¶?|
| ?°ì´???ì„± | 10ì´?| 10ì´?|
| VAE ?ˆë ¨ (10 epochs) | 2?œê°„ | 10ë¶?|
| DiT ?ˆë ¨ (20 epochs) | 4?œê°„ | 30ë¶?|
| SVG ?ì„± | 1ë¶?| 10ì´?|
| **ì´í•©** | ~6?œê°„ | ~40ë¶?|

## ?¯ ë¹ ë¥¸ ?ŒìŠ¤??(GPU ?†ì´)

GPUê°€ ?†ë‹¤ë©????‘ì? ?¤ì •?¼ë¡œ ?ŒìŠ¤??

```bash
# 1. ???ì? ?°ì´??
python scripts/create_dummy_data.py --num_samples 10

# 2. ?‘ì? ëª¨ë¸ (config.py ?˜ì •)
# encoder_d_model = 256
# decoder_d_model = 256
# encoder_layers = 2
# decoder_layers = 2

# 3. ?‘ì? ë°°ì¹˜, ?ì? ?í­
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

# 4. ?ì„±
python scripts/generate.py \
    --vae_checkpoint checkpoints/vae/vpvae_final.pt \
    --dit_checkpoint checkpoints/dit/vsdit_final.pt \
    --prompt "a red circle" \
    --num_samples 2 \
    --ddim_steps 50
```

## ?”§ ì£¼ìš” ?Œë¼ë¯¸í„°

### ?ˆë ¨ ?Œë¼ë¯¸í„°

- `--batch_size`: ë°°ì¹˜ ?¬ê¸° (GPU ë©”ëª¨ë¦¬ì— ?°ë¼ ì¡°ì •)
- `--num_epochs`: ?í­ ??(??ë§ì„?˜ë¡ ì¢‹ìŒ)
- `--lr`: ?™ìŠµë¥?(ê¸°ë³¸ê°? 1e-4)
- `--max_seq_len`: ìµœë? ?œí€€??ê¸¸ì´ (ë©”ëª¨ë¦??í–¥)

### ?ì„± ?Œë¼ë¯¸í„°

- `--cfg_scale`: CFG ê°•ë„ (7-15)
  - ??Œ: ?¤ì–‘???? ?ˆì§ˆ ??
  - ?’ìŒ: ?¤ì–‘???? ?ˆì§ˆ ??
  
- `--ddim_steps`: ?˜í”Œë§??¤í… (50-250)
  - ?ìŒ: ë¹ ë¦„, ?ˆì§ˆ ??
  - ë§ìŒ: ?ë¦¼, ?ˆì§ˆ ??
  
- `--eta`: ?•ë¥ ??(0-1)
  - 0: ê²°ì •??
  - 1: ?•ë¥ ??

## ?’¡ ? ìš©????

### 1. ì²´í¬?¬ì¸???¬ê°œ

```bash
# VAE ?ˆë ¨ ?¬ê°œ
python scripts/train_vae.py \
    --svg_dir data/svgs \
    --resume_from checkpoints/vae/vpvae_epoch10.pt \
    --num_epochs 20
```

### 2. ?ì„± ?ˆì§ˆ ê°œì„ 

```bash
# CFG ?¤ì???ì¡°ì •
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

### 3. ë°°ì¹˜ ?ì„±

```bash
# ?¬ëŸ¬ ?„ë¡¬?„íŠ¸???€???ì„±
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

## ?› ë¬¸ì œ ?´ê²°

### Out of Memory

```bash
# ë°°ì¹˜ ?¬ê¸° ê°ì†Œ
--batch_size 2

# ?œí€€??ê¸¸ì´ ê°ì†Œ
--max_seq_len 256

# Gradient accumulation
--accumulation_steps 4
```

### ?ˆë ¨???ë¦¼

```bash
# Workers ì¦ê?
--num_workers 8

# ?„ë² ??ìºì‹± ?œì„±??
# (dataset.py?ì„œ cache_embeddings=True)

# Mixed precision (PyTorch 2.0+)
--fp16
```

### ?ì„± ê²°ê³¼ê°€ ??ì¢‹ìŒ

1. **??ë§ì? ?ˆë ¨ ?°ì´??*
2. **??ê¸??ˆë ¨** (50+ epochs)
3. **CFG ?¤ì???ì¡°ì •** (7-15)
4. **DDIM ?¤í… ì¦ê?** (100-250)

## ?“š ?¤ìŒ ?¨ê³„

1. ??ë¹ ë¥¸ ?œì‘ ?„ë£Œ
2. ?“– [README.md](README.md) ?„ì²´ ë¬¸ì„œ ?½ê¸°
3. ?¨ ?¤ì œ SVG ?°ì´?°ë¡œ ?ˆë ¨
4. ?”§ ëª¨ë¸ ?„í‚¤?ì²˜ ì»¤ìŠ¤?°ë§ˆ?´ì§•
5. ?“Š ê²°ê³¼ ?œê°??ë°??‰ê?

## ?‰ ì¶•í•˜?©ë‹ˆ??

SVG Fusion???±ê³µ?ìœ¼ë¡??¤í–‰?ˆìŠµ?ˆë‹¤! ?´ì œ ?ìŠ¤?¸ì—??SVGë¥??ì„±?????ˆìŠµ?ˆë‹¤.

??ê¶ê¸ˆ???ì´ ?ˆìœ¼ë©?README.mdë¥?ì°¸ê³ ?˜ì„¸??
