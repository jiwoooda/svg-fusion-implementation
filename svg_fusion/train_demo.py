"""
SVG Fusion 훈련 스크립트 (데모)
실제 사용 시 데이터셋을 준비해야 합니다
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from models import VPVAE, VS_DiT
from utils import DiffusionUtils
import config


class DummyDataset(Dataset):
    """데모용 더미 데이터셋"""
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 더미 SVG 매트릭스
        svg_matrix = torch.randint(0, 256, (512, 14))
        
        # 더미 픽셀 임베딩
        pixel_embedding = torch.randn(512, 384)
        
        # 더미 마스크
        mask = torch.zeros(512, dtype=torch.bool)
        
        # 더미 텍스트 임베딩
        text_embedding = torch.randn(77, 768)
        text_mask = torch.zeros(77, dtype=torch.bool)
        
        return {
            'svg_matrix': svg_matrix,
            'pixel_embedding': pixel_embedding,
            'svg_mask': mask,
            'text_embedding': text_embedding,
            'text_mask': text_mask
        }


def train_vae_demo():
    """VAE 훈련 데모"""
    print("=" * 60)
    print("VAE Training Demo")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 모델 생성
    model = VPVAE(
        num_element_types=config.VAE_CONFIG["num_element_types"],
        num_command_types=config.VAE_CONFIG["num_command_types"],
        element_embed_dim=config.VAE_CONFIG["element_embed_dim"],
        command_embed_dim=config.VAE_CONFIG["command_embed_dim"],
        num_continuous_params=config.VAE_CONFIG["num_continuous_params"],
        pixel_feature_dim=config.VAE_CONFIG["pixel_feature_dim"],
        encoder_d_model=config.VAE_CONFIG["encoder_d_model"],
        decoder_d_model=config.VAE_CONFIG["decoder_d_model"],
        encoder_layers=config.VAE_CONFIG["encoder_layers"],
        decoder_layers=config.VAE_CONFIG["decoder_layers"],
        num_heads=config.VAE_CONFIG["num_heads"],
        latent_dim=config.VAE_CONFIG["latent_dim"],
        max_seq_len=config.VAE_CONFIG["max_seq_len"]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
    
    # 데이터셋 & 데이터로더
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=config.VAE_CONFIG["learning_rate"])
    
    # 손실 함수
    ce_loss = nn.CrossEntropyLoss()
    
    # 훈련 루프 (짧은 데모)
    print("Training for 3 steps (demo)...")
    model.train()
    
    for step, batch in enumerate(dataloader):
        if step >= 3:  # 데모용 3 스텝만
            break
        
        svg_matrix = batch['svg_matrix'].to(device)
        pixel_embedding = batch['pixel_embedding'].to(device)
        svg_mask = batch['svg_mask'].to(device)
        
        # Forward
        element_logits, command_logits, param_logits, mu, log_var = model(
            svg_matrix, pixel_embedding, svg_mask
        )
        
        # 손실 계산 (간단화)
        recon_loss = ce_loss(
            element_logits.reshape(-1, element_logits.size(-1)),
            svg_matrix[:, :, 0].long().reshape(-1)
        )
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        total_loss = recon_loss + 0.1 * kl_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}: Loss={total_loss.item():.4f} "
              f"(Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f})")
    
    # 저장
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(config.CHECKPOINT_DIR, "vpvae_demo.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


def train_dit_demo():
    """DiT 훈련 데모"""
    print("\n" + "=" * 60)
    print("DiT Training Demo")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 모델 생성
    model = VS_DiT(
        latent_dim=config.DIT_CONFIG["latent_dim"],
        hidden_dim=config.DIT_CONFIG["hidden_dim"],
        context_dim=config.DIT_CONFIG["context_dim"],
        num_blocks=config.DIT_CONFIG["num_blocks"],
        num_heads=config.DIT_CONFIG["num_heads"],
        mlp_ratio=config.DIT_CONFIG["mlp_ratio"],
        dropout=config.DIT_CONFIG["dropout"]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
    
    # 확산 파라미터
    betas = DiffusionUtils.get_linear_noise_schedule(
        config.DIT_CONFIG["noise_steps"],
        config.DIT_CONFIG["beta_start"],
        config.DIT_CONFIG["beta_end"]
    )
    diff_params = DiffusionUtils.precompute_diffusion_parameters(betas, device)
    
    # 데이터셋 & 데이터로더
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=config.DIT_CONFIG["learning_rate"])
    
    # 손실 함수
    mse_loss = nn.MSELoss()
    
    # 훈련 루프 (짧은 데모)
    print("Training for 3 steps (demo)...")
    model.train()
    
    for step, batch in enumerate(dataloader):
        if step >= 3:  # 데모용 3 스텝만
            break
        
        # 더미 잠재 벡터 (실제로는 VAE 인코더 출력)
        z0 = torch.randn(4, 512, config.DIT_CONFIG["latent_dim"]).to(device)
        
        text_embedding = batch['text_embedding'].to(device)
        text_mask = batch['text_mask'].to(device)
        
        # 노이즈 추가
        t = torch.randint(0, config.DIT_CONFIG["noise_steps"], (4,), device=device)
        zt, noise = DiffusionUtils.noise_latent(z0, t, diff_params, device)
        
        # 노이즈 예측
        predicted_noise = model(zt, t, text_embedding, text_mask)
        
        # 손실 계산
        loss = mse_loss(predicted_noise, noise)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}: Loss={loss.item():.4f}")
    
    # 저장
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(config.CHECKPOINT_DIR, "vsdit_demo.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SVG Fusion models (demo)")
    parser.add_argument("--model", type=str, choices=["vae", "dit", "both"],
                       default="both", help="Which model to train")
    
    args = parser.parse_args()
    
    if args.model in ["vae", "both"]:
        train_vae_demo()
    
    if args.model in ["dit", "both"]:
        train_dit_demo()
    
    print("\n" + "=" * 60)
    print("Demo training complete!")
    print("Note: This is a demo with dummy data.")
    print("For real training, prepare your SVG dataset.")
    print("=" * 60)
