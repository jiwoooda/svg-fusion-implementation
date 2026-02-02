"""
VP-VAE 훈련 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import os

from models import VPVAE
from utils import SVGDataset, collate_fn
from config import VAEConfig


def train_vae(args):
    """VAE 훈련"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋
    print("Loading dataset...")
    if not os.path.exists(args.svg_dir):
        raise ValueError(f"SVG directory not found: {args.svg_dir}")
    
    dataset = SVGDataset(
        svg_dir=args.svg_dir,
        captions_dict=None,
        max_seq_len=args.max_seq_len,
        dino_model_name=args.dino_model,
        cache_embeddings=True
    )
    
    if len(dataset) == 0:
        raise ValueError("No SVG files found in dataset")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # VAE 모델
    print("Initializing VAE...")
    config = VAEConfig()
    config.max_seq_len = args.max_seq_len
    config.pixel_embed_dim = dataset.dino_embed_dim
    
    vae = VPVAE(
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
        max_seq_len=config.max_seq_len
    ).to(device)
    
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.1
    )
    
    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 훈련 루프
    print("Starting training...")
    vae.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            svg_tensor = batch['svg_tensor'].to(device)  # [B, L, 14]
            pixel_emb = batch['pixel_embedding'].to(device)  # [B, L, D]
            svg_mask = batch['svg_mask'].to(device)  # [B, L]

            # SVG 텐서 분리
            element_ids = svg_tensor[:, :, 0]  # [B, L]
            command_ids = svg_tensor[:, :, 1]  # [B, L]
            continuous_params = svg_tensor[:, :, 2:]  # [B, L, 12]

            # Forward
            outputs = vae(
                svg_element_ids=element_ids,
                svg_command_ids=command_ids,
                svg_continuous_params=continuous_params,
                pixel_features=pixel_emb,
                svg_mask=svg_mask
            )

            # ---- Recon loss: masked MSE on continuous features ----
            predicted = outputs['predicted_features']  # [B, L_out, F]

            # 타겟을 [-1,1]로 정규화
            target = vae.normalize_target(element_ids, command_ids, continuous_params)  # [B, L_tgt, F]

            # effective length
            L_out = predicted.size(1)
            L_tgt = target.size(1)
            effective_len = min(L_out, L_tgt)

            preds_eff = predicted[:, :effective_len, :]
            targets_eff = target[:, :effective_len, :]
            mask_eff = svg_mask[:, :effective_len]  # [B, L]

            # feature valid mask: [B, L, F]
            feature_valid_mask = (~mask_eff).unsqueeze(-1).expand_as(targets_eff).float()
            num_valid = feature_valid_mask.sum() + 1e-9

            # masked MSE
            mse_unreduced = nn.functional.mse_loss(
                preds_eff * feature_valid_mask,
                targets_eff * feature_valid_mask,
                reduction='none'
            )
            mse_recon = mse_unreduced.sum() / num_valid

            # ---- KL loss ----
            kl_loss = outputs['kl_loss']

            # KL weight annealing: linear ramp over anneal_steps to kl_weight_max
            global_step = epoch * len(dataloader) + batch_idx
            anneal_steps = int(args.total_steps * args.kl_anneal_portion)
            if global_step < anneal_steps:
                kl_weight = args.kl_weight_max * global_step / max(1, anneal_steps)
            else:
                kl_weight = args.kl_weight_max

            # ---- Total loss ----
            loss = args.recon_mse_loss_weight * mse_recon + kl_weight * kl_loss
            recon_loss = mse_recon  # for logging
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            # 통계
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'kl_w': f'{kl_weight:.3f}'
            })
        
        scheduler.step()
        
        # Epoch 통계
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Recon: {avg_recon:.4f}")
        print(f"  Avg KL: {avg_kl:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"vpvae_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config.__dict__,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # 최종 모델 저장
    final_path = output_dir / "vpvae_final.pt"
    torch.save({
        'model_state_dict': vae.state_dict(),
        'config': config.__dict__
    }, final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VP-VAE")
    
    # 데이터
    parser.add_argument('--svg_dir', type=str, required=True, help='SVG files directory')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--dino_model', type=str, default='facebook/dinov2-small', help='DINOv2 model')
    
    # 훈련
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--total_steps', type=int, default=15000, help='Total training steps (for KL annealing)')
    parser.add_argument('--kl_anneal_portion', type=float, default=0.5, help='Portion of total steps for KL annealing')
    parser.add_argument('--kl_weight_max', type=float, default=0.5, help='Maximum KL weight')
    parser.add_argument('--recon_mse_loss_weight', type=float, default=1.0, help='Recon MSE loss weight')
    
    # 기타
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default='checkpoints/vae', help='Output directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train_vae(args)
