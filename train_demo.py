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
from config import (
    VAEConfig, DiTConfig, CHECKPOINT_DIR,
    NUM_ELEMENT_TYPES, NUM_COMMAND_TYPES, NUM_CONTINUOUS_PARAMS,
    N_BINS, PAD_IDX, BOS_IDX
)


class DummyDataset(Dataset):
    """데모용 더미 데이터셋"""
    def __init__(self, num_samples=1000, seq_len=512):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        L = self.seq_len

        # 더미 SVG tensor [L, 14] - BOS + content + EOS + PAD 형태
        svg_tensor = torch.zeros(L, 2 + NUM_CONTINUOUS_PARAMS, dtype=torch.long)
        # BOS
        svg_tensor[0, 0] = BOS_IDX
        # content: element ids [1..4], command ids [0..12], params [0..255]
        content_len = L - 2  # BOS + content + EOS
        svg_tensor[1:1+content_len, 0] = torch.randint(1, 5, (content_len,))  # rect/circle/ellipse/path
        svg_tensor[1:1+content_len, 1] = torch.randint(0, NUM_COMMAND_TYPES, (content_len,))
        svg_tensor[1:1+content_len, 2:] = torch.randint(0, N_BINS, (content_len, NUM_CONTINUOUS_PARAMS))
        # EOS
        svg_tensor[-1, 0] = 5  # EOS

        # 더미 픽셀 임베딩
        pixel_embedding = torch.randn(L, 384)

        # 더미 마스크 (모두 valid)
        mask = torch.zeros(L, dtype=torch.bool)

        # 더미 텍스트 임베딩
        text_embedding = torch.randn(77, 768)
        text_mask = torch.zeros(77, dtype=torch.bool)

        return {
            'svg_tensor': svg_tensor,
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

    vae_config = VAEConfig()

    # 모델 생성
    model = VPVAE(
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

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    # 데이터셋 & 데이터로더
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=vae_config.learning_rate)

    # 훈련 루프 (짧은 데모)
    print("Training for 3 steps (demo)...")
    model.train()

    for step, batch in enumerate(dataloader):
        if step >= 3:
            break

        svg_tensor = batch['svg_tensor'].to(device)
        pixel_embedding = batch['pixel_embedding'].to(device)
        svg_mask = batch['svg_mask'].to(device)

        element_ids = svg_tensor[:, :, 0]
        command_ids = svg_tensor[:, :, 1]
        continuous_params = svg_tensor[:, :, 2:]

        # Forward (새 인터페이스: 분리된 인자 + dict 반환)
        outputs = model(
            svg_element_ids=element_ids,
            svg_command_ids=command_ids,
            svg_continuous_params=continuous_params,
            pixel_features=pixel_embedding,
            svg_mask=svg_mask
        )

        # Reconstruction loss
        B, L = element_ids.shape
        valid_mask = ~svg_mask

        # Element CE loss
        elem_loss = nn.functional.cross_entropy(
            outputs['element_logits'].reshape(-1, outputs['element_logits'].size(-1)),
            element_ids.reshape(-1).long(),
            reduction='none'
        )
        elem_loss = (elem_loss * valid_mask.reshape(-1).float()).sum() / valid_mask.sum()

        # Command CE loss
        cmd_loss = nn.functional.cross_entropy(
            outputs['command_logits'].reshape(-1, outputs['command_logits'].size(-1)),
            command_ids.reshape(-1).long(),
            reduction='none'
        )
        cmd_loss = (cmd_loss * valid_mask.reshape(-1).float()).sum() / valid_mask.sum()

        # Param CE loss
        B_p, L_p, P, num_bins = outputs['param_logits'].shape
        param_logits_flat = outputs['param_logits'].reshape(-1, num_bins)
        params_target_flat = continuous_params.reshape(-1).long()
        cont_loss = nn.functional.cross_entropy(
            param_logits_flat, params_target_flat, reduction='none'
        )
        cont_loss = cont_loss.reshape(B, L, P).mean(dim=-1)
        cont_loss = (cont_loss * valid_mask.float()).sum() / valid_mask.sum()

        recon_loss = elem_loss + cmd_loss + cont_loss
        kl_loss = outputs['kl_loss']

        total_loss = recon_loss + 0.1 * kl_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Step {step+1}: Loss={total_loss.item():.4f} "
              f"(Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f})")

    # 저장
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, "vpvae_demo.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


def train_dit_demo():
    """DiT 훈련 데모"""
    print("\n" + "=" * 60)
    print("DiT Training Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    dit_config = DiTConfig()

    # 모델 생성
    model = VS_DiT(
        latent_dim=dit_config.latent_dim,
        hidden_dim=dit_config.hidden_dim,
        context_dim=dit_config.context_dim,
        num_blocks=dit_config.num_blocks,
        num_heads=dit_config.num_heads,
        mlp_ratio=dit_config.mlp_ratio,
        dropout=dit_config.dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    # 확산 파라미터
    betas = DiffusionUtils.get_linear_noise_schedule(
        dit_config.noise_steps,
        dit_config.beta_start,
        dit_config.beta_end
    )
    diff_params = DiffusionUtils.precompute_diffusion_parameters(betas, device)

    # 데이터셋 & 데이터로더
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=dit_config.learning_rate)

    # 손실 함수
    mse_loss = nn.MSELoss()

    # 훈련 루프 (짧은 데모)
    print("Training for 3 steps (demo)...")
    model.train()

    for step, batch in enumerate(dataloader):
        if step >= 3:
            break

        # 더미 잠재 벡터 (실제로는 VAE 인코더 출력)
        z0 = torch.randn(4, 512, dit_config.latent_dim).to(device)

        text_embedding = batch['text_embedding'].to(device)
        text_mask = batch['text_mask'].to(device)

        # 노이즈 추가
        t = torch.randint(0, dit_config.noise_steps, (4,), device=device)
        zt, noise = DiffusionUtils.noise_latent(z0, t, diff_params)

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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, "vsdit_demo.pt")
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
