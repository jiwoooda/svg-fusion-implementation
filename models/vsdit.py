"""
Vector Space Diffusion Transformer (VS-DiT)
텍스트 조건부 잠재 공간 확산 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def modulate(x, shift, scale):
    """AdaLN 변조"""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """타임스텝 임베딩"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Sinusoidal 타임스텝 임베딩"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SelfAttention(nn.Module):
    """Self-Attention with numerical stability"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-Attention for text conditioning"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, context, context_mask=None):
        """
        x: [B, N, dim] - query (latent)
        context: [B, S, dim] - key/value (text)
        context_mask: [B, S] - True=padding
        """
        B, N, C = x.shape
        S = context.shape[1]
        
        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv = self.kv_linear(context).reshape(B, S, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, S)
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VS_DiT_Block(nn.Module):
    """VS-DiT Block with AdaLN modulation"""
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # AdaLN 변조
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        
        # Attention layers
        self.self_attn = SelfAttention(hidden_dim, num_heads)
        self.cross_attn = CrossAttention(hidden_dim, num_heads)
        
        # Feed-forward
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = FeedForward(hidden_dim, mlp_hidden_dim)
    
    def forward(self, x, t_emb, context, context_mask=None):
        """
        x: [B, N, hidden_dim] - latent
        t_emb: [B, hidden_dim] - timestep embedding
        context: [B, S, hidden_dim] - text
        context_mask: [B, S] - padding mask
        """
        # AdaLN 파라미터
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        shift_sa = shift_sa.unsqueeze(1)
        scale_sa = scale_sa.unsqueeze(1)
        gate_sa = gate_sa.unsqueeze(1)
        shift_ff = shift_ff.unsqueeze(1)
        scale_ff = scale_ff.unsqueeze(1)
        gate_ff = gate_ff.unsqueeze(1)
        
        # Self-attention
        x_norm1 = self.norm1(x)
        x_modulated1 = modulate(x_norm1, shift_sa, scale_sa)
        x_sa = self.self_attn(x_modulated1)
        x = x + gate_sa * x_sa
        
        # Cross-attention
        x_norm2 = self.norm2(x)
        x_ca = self.cross_attn(x_norm2, context, context_mask)
        x = x + x_ca
        
        # Feed-forward
        x_norm3 = self.norm3(x)
        x_modulated3 = modulate(x_norm3, shift_ff, scale_ff)
        x_ff = self.mlp(x_modulated3)
        x = x + gate_ff * x_ff
        
        return x


class VS_DiT(nn.Module):
    """Vector Space Diffusion Transformer"""
    def __init__(self, latent_dim, hidden_dim, context_dim,
                 num_blocks, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # 타임스텝 임베더
        self.t_embedder = TimestepEmbedder(hidden_dim)
        
        # 입력 프로젝션
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        
        # 컨텍스트 프로젝션
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # DiT 블록
        self.blocks = nn.ModuleList([
            VS_DiT_Block(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_blocks)
        ])
        
        # 최종 레이어
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        self.final_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """가중치 초기화"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)
    
    def forward(self, z_t, t, context_seq, context_padding_mask=None):
        """
        z_t: [B, N, latent_dim] - noisy latent
        t: [B] - timestep
        context_seq: [B, S, context_dim] - text embedding
        context_padding_mask: [B, S] - padding mask
        """
        B, N, _ = z_t.shape
        
        # 프로젝션
        h = self.proj_in(z_t)
        
        # 타임스텝 임베딩
        t_emb = self.t_embedder(t)
        
        # 컨텍스트 프로젝션
        context = self.context_proj(context_seq)
        
        # DiT 블록 적용
        for block in self.blocks:
            h = block(h, t_emb, context, context_padding_mask)
        
        # 최종 처리
        shift, scale = self.final_adaLN_modulation(t_emb).chunk(2, dim=1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        
        h = self.final_norm(h)
        h = modulate(h, shift, scale)
        epsilon_pred = self.final_proj(h)
        
        return epsilon_pred
