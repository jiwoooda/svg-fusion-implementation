#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/8TB_1/jiwoo/projects/svg_smoke/svg_fusion"
CACHE="/mnt/8TB_1/jiwoo/projects/svg_smoke/precomputed_patch_tokens_data"
VAE_CKPT="/mnt/8TB_1/jiwoo/checkpoints/model_step300.pt"
DIT_CKPT="/mnt/8TB_1/jiwoo/checkpoints/dit_v2/vsdit_final.pt"
OUT="/mnt/8TB_1/jiwoo/outputs"

cd "$ROOT"

echo "== [1] Precomputed cache (one sample) =="
python - <<'EOF'
import torch, glob
p = sorted(glob.glob("/mnt/8TB_1/jiwoo/projects/svg_smoke/precomputed_patch_tokens_data/*.pt"))[0]
d = torch.load(p, map_location="cpu")
print("keys:", d.keys())
print("full_svg_matrix_content:", d["full_svg_matrix_content"].shape, d["full_svg_matrix_content"].dtype)
print("final_pixel_cls_token:", d["final_pixel_cls_token"].shape, d["final_pixel_cls_token"].dtype)
print("caption:", d.get("caption"))
EOF

echo "== [2] VAE output shape/range =="
python - <<'EOF'
import torch, glob
from models import VPVAE
from config import VAEConfig
ckpt = torch.load("/mnt/8TB_1/jiwoo/checkpoints/model_step300.pt", map_location="cpu")
cfg = VAEConfig()
for k,v in ckpt.get("config", {}).items():
    if hasattr(cfg, k): setattr(cfg, k, v)
vae = VPVAE(
    num_element_types=cfg.num_element_types,
    num_command_types=cfg.num_command_types,
    element_embed_dim=cfg.element_embed_dim,
    command_embed_dim=cfg.command_embed_dim,
    num_continuous_params=cfg.num_continuous_params,
    pixel_feature_dim=cfg.pixel_embed_dim,
    encoder_d_model=cfg.encoder_d_model,
    decoder_d_model=cfg.decoder_d_model,
    encoder_layers=cfg.encoder_layers,
    decoder_layers=cfg.decoder_layers,
    num_heads=cfg.num_heads,
    latent_dim=cfg.latent_dim,
    max_seq_len=cfg.max_seq_len
)
vae.load_state_dict(ckpt["model_state_dict"])
vae.eval()

p = sorted(glob.glob("/mnt/8TB_1/jiwoo/projects/svg_smoke/precomputed_patch_tokens_data/*.pt"))[0]
d = torch.load(p, map_location="cpu")
svg = d["full_svg_matrix_content"].unsqueeze(0)
pix = d["final_pixel_cls_token"].view(1,1,-1).repeat(1, svg.size(1), 1)
mask = torch.zeros((1, svg.size(1)), dtype=torch.bool)

out = vae(
    svg_element_ids=svg[:,:,0],
    svg_command_ids=svg[:,:,1],
    svg_continuous_params=svg[:,:,2:],
    pixel_features=pix,
    svg_mask=mask
)

pred = out["predicted_features"]
print("predicted_features:", pred.shape, float(pred.min()), float(pred.max()))
print("kl_loss:", float(out["kl_loss"]))
EOF

echo "== [3] DiT output shape =="
python - <<'EOF'
import torch
from models import VSDiT
from config import DiTConfig
from utils.diffusion import DiffusionUtils

ckpt = torch.load("/mnt/8TB_1/jiwoo/checkpoints/dit_v2/vsdit_final.pt", map_location="cpu")
cfg = DiTConfig()
for k,v in ckpt.get("config", {}).items():
    if hasattr(cfg, k): setattr(cfg, k, v)

model = VSDiT(
    latent_dim=cfg.latent_dim,
    hidden_dim=cfg.hidden_dim,
    context_dim=cfg.context_dim,
    num_blocks=cfg.num_blocks,
    num_heads=cfg.num_heads,
    mlp_ratio=cfg.mlp_ratio,
    dropout=cfg.dropout
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

B, L = 2, 1024
z0 = torch.randn(B, L, cfg.latent_dim)
t = torch.randint(0, cfg.noise_steps, (B,))
betas = DiffusionUtils.get_linear_noise_schedule(cfg.noise_steps, cfg.beta_start, cfg.beta_end)
diff = DiffusionUtils.precompute_diffusion_parameters(betas, device="cpu")
zt, noise = DiffusionUtils.noise_latent(z0, t, diff)
context = torch.randn(B, 77, cfg.context_dim)
mask = torch.zeros(B, 77, dtype=torch.bool)

pred = model(zt, t, context, mask)
print("pred_noise:", pred.shape, float(pred.mean()), float(pred.std()))
EOF

echo "== [4] Outputs =="
ls "$OUT" | head || true
ls "$OUT" | grep dit_sample | head || true
