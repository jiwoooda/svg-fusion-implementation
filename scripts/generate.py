import argparse
from pathlib import Path

import torch
import cairosvg

from models import VPVAE
from config import VAEConfig
from utils.tensorsvg import TensorToSVGHybrid
from utils.hybrid_utils import compute_actual_len


def safe_write_svg(converter, tensor, out_path, actual_len=None):
    converter.tensor_to_svg_file(tensor=tensor, output_file=str(out_path), actual_len=actual_len)
    try:
        cairosvg.svg2png(url=str(out_path))
        return True
    except Exception:
        return False


def load_precomputed(precomputed_dir: str, max_files: int):
    files = sorted(Path(precomputed_dir).glob("*.pt"))
    if max_files and max_files > 0:
        files = files[:max_files]
    return files


def main():
    parser = argparse.ArgumentParser(description="Generate reconstructions and samples from a VAE")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--precomputed_dir", type=str, required=True, help="Cache directory")
    parser.add_argument("--num_eval", type=int, default=10, help="Number of reconstructions")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    config_dict = ckpt.get("config", {})
    config = VAEConfig()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    model = VPVAE(
        num_element_types=config.num_element_types,
        num_command_types=config.num_command_types,
        element_embed_dim=config.element_embed_dim,
        command_embed_dim=config.command_embed_dim,
        num_continuous_params=config.num_continuous_params,
        pixel_feature_dim=config.pixel_embed_dim,
        encoder_d_model=config.encoder_d_model,
        decoder_d_model=config.decoder_d_model,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        num_heads=config.num_heads,
        latent_dim=config.latent_dim,
        max_seq_len=config.max_seq_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step_tag = "step"
    if "step" in ckpt:
        step_tag = f"step{ckpt['step']}"
    elif "model_step" in Path(args.ckpt_path).stem:
        step_tag = Path(args.ckpt_path).stem.replace("model_", "")

    out_dir = Path(args.out_dir)
    recon_dir = out_dir / "recon" / step_tag
    sample_dir = out_dir / "samples" / step_tag
    recon_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    converter = TensorToSVGHybrid()

    # Reconstructions
    files = load_precomputed(args.precomputed_dir, args.num_eval)
    if not files:
        print("No precomputed files found.")
        return

    bad = 0
    with torch.no_grad():
        for i, fpath in enumerate(files):
            payload = torch.load(fpath, map_location="cpu")
            svg_tensor = payload["full_svg_matrix_content"].long()
            pixel_cls = payload["final_pixel_cls_token"].float().view(1, -1)

            seq_len = svg_tensor.shape[0]
            pixel_seq = pixel_cls.repeat(seq_len, 1).unsqueeze(0).to(device)
            svg_tensor_b = svg_tensor.unsqueeze(0).to(device)
            svg_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)

            element_ids = svg_tensor_b[:, :, 0]
            command_ids = svg_tensor_b[:, :, 1]
            continuous_params = svg_tensor_b[:, :, 2:]

            outputs = model(
                svg_element_ids=element_ids,
                svg_command_ids=command_ids,
                svg_continuous_params=continuous_params,
                pixel_features=pixel_seq,
                svg_mask=svg_mask,
            )

            pred = outputs["predicted_features"]
            elem_ids, cmd_ids, params = model.denormalize_output(pred)
            recon_tensor = torch.cat(
                [elem_ids.unsqueeze(-1), cmd_ids.unsqueeze(-1), params], dim=-1
            ).squeeze(0).cpu()

            actual_len = compute_actual_len(svg_tensor)
            orig_path = recon_dir / f"original_{i:03d}.svg"
            recon_path = recon_dir / f"recon_{i:03d}.svg"

            ok1 = safe_write_svg(converter, svg_tensor, orig_path, actual_len=actual_len)
            ok2 = safe_write_svg(converter, recon_tensor, recon_path, actual_len=actual_len)
            if not ok1 or not ok2:
                bad += 1

    # Random sampling
    with torch.no_grad():
        z = torch.randn(
            args.num_samples, config.max_seq_len, config.latent_dim, device=device
        )
        pred = model.decode(z)
        elem_ids, cmd_ids, params = model.denormalize_output(pred)
        sample_tensor = torch.cat(
            [elem_ids.unsqueeze(-1), cmd_ids.unsqueeze(-1), params], dim=-1
        ).cpu()

        for i in range(args.num_samples):
            out_path = sample_dir / f"sample_{i:03d}.svg"
            ok = safe_write_svg(converter, sample_tensor[i], out_path)
            if not ok:
                bad += 1

    print(
        f"Done. recon_dir={recon_dir} sample_dir={sample_dir} bad_svgs={bad}"
    )


if __name__ == "__main__":
    main()
