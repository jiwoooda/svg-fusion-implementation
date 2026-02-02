import argparse
from pathlib import Path

import torch
import cairosvg
from transformers import CLIPTokenizer, CLIPTextModel

from models import VPVAE, VSDiT
from config import VAEConfig, DiTConfig
from utils.diffusion import DiffusionUtils
from utils.tensorsvg import TensorToSVGHybrid
from utils.hybrid_utils import compute_actual_len


def safe_write_svg(converter, tensor, out_path, actual_len=None):
    converter.tensor_to_svg_file(tensor=tensor, output_file=str(out_path), actual_len=actual_len)
    try:
        cairosvg.svg2png(url=str(out_path))
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate SVGs with DiT + VAE")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--dit_ckpt", type=str, required=True, help="DiT checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples")
    parser.add_argument("--ddim_steps", type=int, default=100, help="DDIM steps")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load VAE
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae_config = VAEConfig()
    for k, v in vae_ckpt.get("config", {}).items():
        if hasattr(vae_config, k):
            setattr(vae_config, k, v)

    vae = VPVAE(
        num_element_types=vae_config.num_element_types,
        num_command_types=vae_config.num_command_types,
        element_embed_dim=vae_config.element_embed_dim,
        command_embed_dim=vae_config.command_embed_dim,
        num_continuous_params=vae_config.num_continuous_params,
        pixel_feature_dim=vae_config.pixel_embed_dim,
        encoder_d_model=vae_config.encoder_d_model,
        decoder_d_model=vae_config.decoder_d_model,
        encoder_layers=vae_config.encoder_layers,
        decoder_layers=vae_config.decoder_layers,
        num_heads=vae_config.num_heads,
        latent_dim=vae_config.latent_dim,
        max_seq_len=vae_config.max_seq_len,
    ).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()

    # Load DiT
    dit_ckpt = torch.load(args.dit_ckpt, map_location=device)
    dit_config = DiTConfig()
    for k, v in dit_ckpt.get("config", {}).items():
        if hasattr(dit_config, k):
            setattr(dit_config, k, v)

    dit = VSDiT(
        latent_dim=dit_config.latent_dim,
        hidden_dim=dit_config.hidden_dim,
        context_dim=dit_config.context_dim,
        num_blocks=dit_config.num_blocks,
        num_heads=dit_config.num_heads,
        mlp_ratio=dit_config.mlp_ratio,
        dropout=dit_config.dropout,
    ).to(device)
    dit.load_state_dict(dit_ckpt["model_state_dict"])
    dit.eval()

    # CLIP text encoder
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    clip_text_encoder = CLIPTextModel.from_pretrained(args.clip_model).to(device)
    clip_text_encoder.eval()

    # Diffusion params
    betas = DiffusionUtils.get_linear_noise_schedule(
        dit_config.noise_steps,
        dit_config.beta_start,
        dit_config.beta_end,
    )
    diff_params = DiffusionUtils.precompute_diffusion_parameters(betas, device)

    # Text context
    with torch.no_grad():
        text_inputs = clip_tokenizer(
            [args.prompt] * args.num_samples,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        text_outputs = clip_text_encoder(**text_inputs)
        cond_context = text_outputs.last_hidden_state
        cond_mask = ~(text_inputs.attention_mask.bool())

    # DDIM sample
    latent_shape = (args.num_samples, vae_config.max_seq_len, dit_config.latent_dim)
    with torch.no_grad():
        z_t = DiffusionUtils.ddim_sample(
            model=dit,
            shape=latent_shape,
            context_seq=cond_context,
            context_padding_mask=cond_mask,
            diff_params=diff_params,
            num_timesteps=dit_config.noise_steps,
            device=device,
            cfg_scale=args.cfg_scale,
            eta=args.eta,
            clip_tokenizer=clip_tokenizer,
            clip_model=clip_text_encoder,
            ddim_steps=args.ddim_steps,
        )

        pred = vae.decode(z_t)
        elem_ids, cmd_ids, params = vae.denormalize_output(pred)
        hybrid = torch.cat([elem_ids.unsqueeze(-1), cmd_ids.unsqueeze(-1), params], dim=-1).cpu()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    converter = TensorToSVGHybrid()

    bad = 0
    for i in range(args.num_samples):
        out_path = out_dir / f"dit_sample_{i:03d}.svg"
        ok = safe_write_svg(converter, hybrid[i], out_path, actual_len=compute_actual_len(hybrid[i]))
        if not ok:
            bad += 1

    print(f"Done. out_dir={out_dir} bad_svgs={bad}")


if __name__ == "__main__":
    main()
