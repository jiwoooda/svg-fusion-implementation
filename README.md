# SVG Fusion

VP-VAE(ë²¡í„°-?½ì? VAE)?€ VS-DiT(?”í“¨???¸ëœ?¤í¬ë¨?ë¥??´ìš©??ë²¡í„° ê·¸ë˜???ì„± ?„ë¡œ?íŠ¸?…ë‹ˆ??

## êµ¬ì„± ?”ì•½

- VP-VAE: SVG ? í° + DINOv2 ?½ì? ?„ë² ?©ì„ ? ì¬ ?œí€€?¤ë¡œ ?¸ì½”?©í•˜ê³??°ì†??SVG ?¼ì²˜ë¥?ë³µì›?©ë‹ˆ??
- VS-DiT: ? ì¬ ê³µê°„?ì„œ CLIP ?ìŠ¤??ì¡°ê±´???¬ìš©??epsilon ?ˆì¸¡ ?”í“¨?„ì„ ?™ìŠµ?©ë‹ˆ??
- VAE/DiT ?™ìŠµ ?¤í¬ë¦½íŠ¸?€ ?°ëª¨ ?¤í¬ë¦½íŠ¸ê°€ ?¬í•¨?©ë‹ˆ??

## ?„ë¡œ?íŠ¸ êµ¬ì¡°

```
svg_fusion/
  models/
    vpvae.py              VP-VAE ?¸ì½”???”ì½”??    vsdit.py              VS-DiT ë¸”ë¡ ë°?ëª¨ë¸
  utils/
    diffusion.py          ?”í“¨??? í‹¸ (?¤ì?ì¤? noise_latent, DDIM)
    dataset.py            SVGDataset, collate_fn
    svg_parser.py         SVG ?Œì‹± ë°??ì„œ??    tensorsvg.py          Tensor -> SVG ë³€??? í‹¸
    hybrid_utils.py       ?˜ì´ë¸Œë¦¬???”ì½”???¬í¼
  config.py               ëª¨ë¸/?™ìŠµ ?¤ì •
  preprocess.py           ?„ì²˜ë¦?ìºì‹œ ?ì„± (?¤ëª¨???ŒìŠ¤?¸ìš©)
  train.py                ?¤ëª¨???ŒìŠ¤?¸ìš© VAE ?™ìŠµ
  generate.py             ?¬êµ¬???œë¤ ?˜í”Œ ?ì„± (?¤ëª¨???ŒìŠ¤?¸ìš©)
  train_vae.py            VP-VAE ?™ìŠµ
  train_dit.py            VS-DiT ?™ìŠµ
  train_demo.py           ?”ë? ?°ì´???°ëª¨ ?™ìŠµ
  create_dummy_data.py    ?©ì„± SVG ?°ì´???ì„±
  prepare_latents.py      ? ì¬ê°??„ì²˜ë¦??¬í¼
  datasetpreparation_v5.py  ?°ì´?°ì…‹ ì¤€ë¹??¤í¬ë¦½íŠ¸
  requirements.txt        ?Œì´???˜ì¡´??```

## ë¹ ë¥¸ ?œì‘ (?¤ëª¨???ŒìŠ¤???Œì´?„ë¼??

?˜ì¡´???¤ì¹˜:

```bash
pip install -r requirements.txt
```

?”ë? ?°ì´???ì„±:

```bash
python scripts/create_dummy_data.py --output_dir data/svgs --num_samples 50
```

?„ì²˜ë¦?(SVG -> ìºì‹œ):

```bash
python scripts/preprocess.py \
  --svg_dir data/svgs \
  --pattern "*.svg" \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --max_files 300
```

?¤ëª¨???ŒìŠ¤?¸ìš© VAE ?™ìŠµ:

```bash
python scripts/train.py \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --steps 2000 \
  --batch_size 4 \
  --device cpu \
  --ckpt_out ./checkpoints
```

?¬êµ¬???˜í”Œ ?ì„±:

```bash
python scripts/generate.py \
  --ckpt_path ./checkpoints/model_step2000.pt \
  --precomputed_dir ./precomputed_patch_tokens_data \
  --num_eval 10 \
  --num_samples 10 \
  --out_dir ./outputs
```

## ê¸°ì¡´ ?™ìŠµ (? íƒ)

VP-VAE ?™ìŠµ:

```bash
python scripts/train_vae.py \
  --svg_dir data/svgs \
  --batch_size 4 \
  --num_epochs 20 \
  --output_dir checkpoints/vae
```

VS-DiT ?™ìŠµ:

```bash
python scripts/train_dit.py \
  --svg_dir data/svgs \
  --vae_checkpoint checkpoints/vae/vpvae_final.pt \
  --batch_size 4 \
  --num_epochs 30 \
  --output_dir checkpoints/dit
```

SVG ?ì„±:

```bash
python scripts/generate.py \
  --vae_checkpoint checkpoints/vae/vpvae_final.pt \
  --dit_checkpoint checkpoints/dit/vsdit_final.pt \
  --prompt "a red circle" \
  --num_samples 4 \
  --output_dir outputs
```

## Loss ê·œê²© (?„ì¬ ì½”ë“œ ê¸°ì?)

### VP-VAE (train_vae.py)

- ?”ì½”??ì¶œë ¥: `predicted_features`??`tanh`ë¡?[-1, 1] ë²”ìœ„.
- ?€ê²? `normalize_target(...)`ê°€ ?°ì† ?¼ì²˜ë¥?[-1, 1]ë¡?ë³€??
- ë§ˆìŠ¤??MSE:
  - `effective_len = min(L_out, L_tgt)`
  - `feature_valid_mask = (~svg_mask).unsqueeze(-1)`ë¥?`[B,L,F]`ë¡??•ì¥
  - `mse_recon = sum(mse(pred*mask, tgt*mask)) / (sum(mask) + 1e-9)`
- KL:
  - `logvar`??`exp` ê³„ì‚°?ì„œ `max=80.0`?¼ë¡œ clamp
  - position-wise KL??`svg_mask`ë¡?ë§ˆìŠ¤????valid ?‰ê· 
- ìµœì¢…:
  - `loss = recon_mse_loss_weight * mse_recon + kl_weight * kl_loss`
  - KL weight??? í˜•?¼ë¡œ `kl_weight_max`ê¹Œì? ì¦ê?

### VS-DiT (train_dit.py)

- `t ~ Uniform(0..T-1)` ?˜í”Œ.
- `(z_t, noise) = DiffusionUtils.noise_latent(z0, t, diff_params)`.
- CLIP ì»¨í…?¤íŠ¸??`last_hidden_state`?€ padding mask(`True = pad`) ?¬ìš©.
- CFG dropout?€ ?˜í”Œ ?¨ìœ„ë¡?conditional/unconditional ì»¨í…?¤íŠ¸?€ ë§ˆìŠ¤?¬ë? ?¤ì™‘.
- ?ˆì¸¡: `predicted_noise = dit(z_t, t, context, context_mask)`.
- ?ì‹¤: `MSE(predicted_noise, noise)` (reduction="mean").

## ?¤ì •

`config.py` ì°¸ê³ :
- `VAEConfig`: latent ?¬ê¸°, ?¸ì½”???”ì½”??ê¹Šì´, ?œí€€??ê¸¸ì´ ??- `DiTConfig`: diffusion steps, hidden size, CLIP ì»¨í…?¤íŠ¸ ì°¨ì›, CFG dropout ??
## ?¤ëª¨???ŒìŠ¤???±ê³µ ì¡°ê±´

- ?„ì²˜ë¦?ìºì‹œê°€ ?ì„±??(.pt)
- DataLoaderê°€ 1ë°°ì¹˜ ?´ìƒ ?µê³¼
- ?™ìŠµ??300 step ?´ìƒ ì§„í–‰?˜ê³  loss ë¡œê·¸ê°€ ì¶œë ¥??- recon SVGê°€ 5ê°??´ìƒ ?€?¥ë¨
- (ê°€?¥í•˜ë©? random ?˜í”Œ SVGê°€ 5ê°??´ìƒ ?€?¥ë¨
- ?ì„±??SVGê°€ cairosvgë¡??Œë” ê°€?¥í•œ ?Œì¼

## ì°¸ê³ 

- `train_demo.py`???”ë? ?°ì´?°ë¡œ ?Œê·œëª??°ëª¨ ?™ìŠµ???˜í–‰?©ë‹ˆ??
- `prepare_latents.py`???”í“¨???™ìŠµ??VAE ? ì¬ê°’ì„ ë¯¸ë¦¬ ê³„ì‚°?????¬ìš©?©ë‹ˆ??
- `generate.py`??DDIM ?˜í”Œë§ê³¼ CLIP ?ìŠ¤??ì¡°ê±´???¬ìš©?©ë‹ˆ??
