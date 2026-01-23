"""
확산 모델 유틸리티
노이즈 스케줄, 샘플링 등
"""

import torch
import math
from tqdm import tqdm


class DiffusionUtils:
    """확산 모델 유틸리티"""
    
    @staticmethod
    def get_linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
        """선형 노이즈 스케줄 생성"""
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    
    @staticmethod
    def precompute_diffusion_parameters(betas, device):
        """확산 파라미터 사전 계산"""
        betas = betas.to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device),
            alphas_cumprod[:-1]
        ])
        
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        return {
            "betas": betas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev
        }
    
    @staticmethod
    def noise_latent(z0, t, diff_params, device):
        """
        잠재 벡터에 노이즈 추가
        
        Args:
            z0: [B, N, D] - 원본 잠재 벡터
            t: [B] - 타임스텝
            diff_params: 확산 파라미터
        
        Returns:
            zt: 노이즈가 추가된 잠재 벡터
            epsilon: 추가된 노이즈
        """
        z0 = z0.to(device)
        t = t.to(device)
        
        sqrt_alpha_bar = diff_params["sqrt_alphas_cumprod"][t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = diff_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1)
        
        epsilon = torch.randn_like(z0, device=device)
        zt = sqrt_alpha_bar * z0 + sqrt_one_minus_alpha_bar * epsilon
        
        return zt, epsilon
    
    @staticmethod
    @torch.no_grad()
    def ddim_sample(model, shape, context_seq, context_padding_mask,
                   diff_params, num_timesteps, device, cfg_scale=7.0,
                   eta=0.0, clip_tokenizer=None, clip_model=None,
                   ddim_steps=100):
        """
        DDIM 샘플링
        
        Args:
            model: VS-DiT 모델
            shape: [B, N, D] - 샘플 형태
            context_seq: [B, S, D_context] - 텍스트 임베딩
            context_padding_mask: [B, S] - 패딩 마스크
            diff_params: 확산 파라미터
            num_timesteps: 전체 타임스텝 수
            device: 디바이스
            cfg_scale: CFG 스케일
            eta: DDIM eta (0=결정적)
            ddim_steps: DDIM 샘플링 스텝 수
        
        Returns:
            z_0: 생성된 잠재 벡터
        """
        model.eval()
        batch_size = shape[0]
        
        # 초기 노이즈
        z_t = torch.randn(shape, device=device)
        
        # 컨텍스트를 디바이스로 이동
        context_seq = context_seq.to(device)
        if context_padding_mask is not None:
            context_padding_mask = context_padding_mask.to(device)
        
        # Unconditional 컨텍스트 생성
        text_seq_len = context_seq.size(1)
        empty_text_inputs = clip_tokenizer(
            [""] * batch_size,
            padding='max_length',
            max_length=text_seq_len,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            empty_outputs = clip_model(**empty_text_inputs)
            uncond_context = empty_outputs.last_hidden_state
            uncond_mask = ~(empty_text_inputs.attention_mask.bool())
        
        # 타임스텝 결정
        step_ratio = num_timesteps // ddim_steps
        timesteps = list(range(0, num_timesteps, step_ratio))
        timesteps = list(reversed(timesteps))
        
        # 샘플링 루프
        for i, t_val in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            
            # t_prev 결정
            if i < len(timesteps) - 1:
                t_prev_val = timesteps[i + 1]
            else:
                t_prev_val = -1
            
            # Alpha 값
            alpha_bar_t = diff_params["alphas_cumprod"][t_val]
            if t_prev_val >= 0:
                alpha_bar_prev = diff_params["alphas_cumprod"][t_prev_val]
            else:
                alpha_bar_prev = 1.0
            
            # CFG로 노이즈 예측
            if cfg_scale > 1.0:
                z_t_combined = torch.cat([z_t, z_t], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                context_combined = torch.cat([context_seq, uncond_context], dim=0)
                mask_combined = torch.cat([context_padding_mask, uncond_mask], dim=0)
                
                eps_combined = model(z_t_combined, t_combined, context_combined, mask_combined)
                eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
                epsilon_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                epsilon_pred = model(z_t, t, context_seq, context_padding_mask)
            
            # x0 예측
            sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = math.sqrt(1 - alpha_bar_t)
            
            pred_x0 = (z_t - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t
            
            # Sigma 계산
            if eta > 0 and i < len(timesteps) - 1:
                sigma = eta * math.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t) *
                    (1 - alpha_bar_t / alpha_bar_prev)
                )
            else:
                sigma = 0.0
            
            # 방향 계산
            if i < len(timesteps) - 1:
                direction = math.sqrt(1 - alpha_bar_prev - sigma**2) * epsilon_pred
            else:
                direction = torch.zeros_like(epsilon_pred)
            
            # 업데이트
            z_t = math.sqrt(alpha_bar_prev) * pred_x0 + direction
            
            # 노이즈 추가 (eta > 0인 경우)
            if sigma > 0:
                noise = torch.randn_like(z_t)
                z_t = z_t + sigma * noise
        
        return z_t
