"""
Pretrained VPVAE encoder로 결정적 latent (z = mu) 추출 및 저장

Usage:
    python prepare_latents.py \
        --svg_dir ./data/svgs \
        --vae_checkpoint ./checkpoints/vpvae_best.pt \
        --output_dir ./data/latents \
        --max_seq_len 1024 \
        --batch_size 16
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import VPVAE
from utils import SVGDataset, collate_fn
from config import VAEConfig


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 시드 고정
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # 데이터셋
    print("Loading dataset...")
    dataset = SVGDataset(
        svg_dir=args.svg_dir,
        max_seq_len=args.max_seq_len,
        dino_model_name=args.dino_model,
        cache_embeddings=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print(f"Dataset size: {len(dataset)}")

    # VAE 로드
    print("Loading VAE encoder...")
    vae_config = VAEConfig()
    vae_config.max_seq_len = args.max_seq_len
    vae_config.pixel_embed_dim = dataset.dino_embed_dim

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
        max_seq_len=vae_config.max_seq_len
    ).to(device)

    if args.vae_checkpoint:
        checkpoint = torch.load(args.vae_checkpoint, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VAE from: {args.vae_checkpoint}")
    else:
        print("Warning: No VAE checkpoint provided, using random initialization")

    vae.eval()

    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Latent 추출
    print("Extracting latents (z = mu, deterministic)...")
    all_latents = []
    all_masks = []
    all_filenames = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            svg_tensor = batch['svg_tensor'].to(device)
            pixel_emb = batch['pixel_embedding'].to(device)
            svg_mask = batch['svg_mask'].to(device)

            element_ids = svg_tensor[:, :, 0]
            command_ids = svg_tensor[:, :, 1]
            continuous_params = svg_tensor[:, :, 2:]

            # z = mu (결정적, 샘플링 없음)
            z_mu = vae.encode(
                svg_element_ids=element_ids,
                svg_command_ids=command_ids,
                svg_continuous_params=continuous_params,
                pixel_features=pixel_emb,
                svg_mask=svg_mask,
                deterministic=True
            )

            all_latents.append(z_mu.cpu())
            all_masks.append(svg_mask.cpu())
            all_filenames.extend(batch['filename'])

    # 저장
    latents = torch.cat(all_latents, dim=0)
    masks = torch.cat(all_masks, dim=0)

    save_path = output_dir / 'latents.pt'
    torch.save({
        'latents': latents,
        'masks': masks,
        'filenames': all_filenames,
        'config': {
            'latent_dim': vae_config.latent_dim,
            'max_seq_len': args.max_seq_len,
            'deterministic': True
        }
    }, save_path)

    print(f"Saved latents to: {save_path}")
    print(f"  Shape: {latents.shape}")
    print(f"  Masks: {masks.shape}")
    print(f"  Files: {len(all_filenames)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract deterministic latents from VAE")
    parser.add_argument('--svg_dir', type=str, required=True, help='SVG 파일 디렉토리')
    parser.add_argument('--vae_checkpoint', type=str, default=None, help='VAE 체크포인트 경로')
    parser.add_argument('--output_dir', type=str, default='./data/latents', help='출력 디렉토리')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='최대 시퀀스 길이')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dino_model', type=str, default='facebook/dinov2-small')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')

    args = parser.parse_args()
    main(args)
