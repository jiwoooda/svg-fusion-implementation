"""
VS-DiT 훈련 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

from models import VPVAE, VSDiT
from utils import SVGDataset, collate_fn, DiffusionUtils
from config import VAEConfig, DiTConfig


def train_dit(args):
    """DiT 훈련"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋
    print("Loading dataset...")
    dataset = SVGDataset(
        svg_dir=args.svg_dir,
        captions_dict=None,
        max_seq_len=args.max_seq_len,
        dino_model_name=args.dino_model,
        cache_embeddings=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # VAE 로드 (인코더만 사용)
    print("Loading VAE encoder...")
    vae_config = VAEConfig()
    vae_config.max_seq_len = args.max_seq_len
    vae_config.pixel_embed_dim = dataset.dino_embed_dim
    
    vae = VPVAE(vae_config).to(device)
    
    if args.vae_checkpoint:
        checkpoint = torch.load(args.vae_checkpoint, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VAE from: {args.vae_checkpoint}")
    else:
        print("Warning: No VAE checkpoint provided, using random initialization")
    
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    # CLIP 텍스트 인코더
    print("Loading CLIP text encoder...")
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    clip_text_encoder = CLIPTextModel.from_pretrained(args.clip_model).to(device)
    clip_text_encoder.eval()
    for param in clip_text_encoder.parameters():
        param.requires_grad = False
    
    # DiT 모델
    print("Initializing DiT...")
    dit_config = DiTConfig()
    dit_config.latent_dim = vae_config.latent_dim
    dit_config.context_dim = clip_text_encoder.config.hidden_size
    
    dit = VSDiT(dit_config).to(device)
    
    total_params = sum(p.numel() for p in dit.parameters())
    print(f"DiT parameters: {total_params:,}")
    
    # Diffusion 유틸리티
    diffusion = DiffusionUtils(
        noise_steps=dit_config.noise_steps,
        beta_start=dit_config.beta_start,
        beta_end=dit_config.beta_end
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        dit.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
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
    dit.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            svg_tensor = batch['svg_tensor'].to(device)
            pixel_emb = batch['pixel_embedding'].to(device)
            svg_mask = batch['svg_mask'].to(device)
            captions = batch['caption']
            
            # 텍스트 임베딩
            with torch.no_grad():
                text_inputs = clip_tokenizer(
                    captions,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).to(device)
                
                text_outputs = clip_text_encoder(**text_inputs)
                text_context = text_outputs.last_hidden_state  # [B, seq_len, D]
            
            # VAE 인코딩
            with torch.no_grad():
                element_ids = svg_tensor[:, :, 0]
                command_ids = svg_tensor[:, :, 1]
                continuous_params = svg_tensor[:, :, 2:]
                
                latents = vae.encode(
                    svg_element_ids=element_ids,
                    svg_command_ids=command_ids,
                    svg_continuous_params=continuous_params,
                    pixel_features=pixel_emb,
                    svg_mask=svg_mask
                )  # [B, L, latent_dim]
            
            # Classifier-Free Guidance (CFG) dropout
            if torch.rand(1).item() < args.cfg_dropout:
                # 25% 확률로 텍스트 컨텍스트를 0으로
                text_context = torch.zeros_like(text_context)
            
            # Timestep 샘플링
            t = torch.randint(0, dit_config.noise_steps, (latents.size(0),), device=device)
            
            # 노이즈 추가
            noise = torch.randn_like(latents)
            noisy_latents = diffusion.noise_latent(latents, noise, t)
            
            # 노이즈 예측
            predicted_noise = dit(
                latent=noisy_latents,
                timestep=t,
                context=text_context,
                latent_mask=svg_mask
            )
            
            # Loss (MSE)
            loss = nn.functional.mse_loss(predicted_noise, noise, reduction='none')
            
            # 마스크 적용
            loss = loss * (~svg_mask).unsqueeze(-1).float()
            loss = loss.sum() / (~svg_mask).sum()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"vsdit_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': dit.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': dit_config.__dict__,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # 최종 모델 저장
    final_path = output_dir / "vsdit_final.pt"
    torch.save({
        'model_state_dict': dit.state_dict(),
        'config': dit_config.__dict__
    }, final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VS-DiT")
    
    # 데이터
    parser.add_argument('--svg_dir', type=str, required=True, help='SVG files directory')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--dino_model', type=str, default='facebook/dinov2-small', help='DINOv2 model')
    
    # 모델
    parser.add_argument('--vae_checkpoint', type=str, required=True, help='VAE checkpoint path')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model')
    
    # 훈련
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--cfg_dropout', type=float, default=0.25, help='CFG dropout probability')
    
    # 기타
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default='checkpoints/dit', help='Output directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train_dit(args)
