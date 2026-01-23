"""
Vector-Pixel VAE (VP-VAE)
SVG 벡터 표현과 픽셀 표현을 융합하는 VAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        
        # 프로젝션
        Q = self.w_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 어텐션
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        # 출력
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(output)


class TransformerBlock(nn.Module):
    """트랜스포머 블록"""
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class VPVAEEncoder(nn.Module):
    """VP-VAE 인코더 - SVG와 픽셀 정보 융합"""
    def __init__(self, num_element_types, num_command_types,
                 element_embed_dim, command_embed_dim,
                 num_continuous_params, pixel_feature_dim,
                 d_model, num_layers, num_heads, latent_dim):
        super().__init__()
        
        # SVG 임베딩
        self.element_embedding = nn.Embedding(num_element_types, element_embed_dim)
        self.command_embedding = nn.Embedding(num_command_types, command_embed_dim)
        self.param_embedding = nn.Embedding(256, 64)  # 양자화된 파라미터
        
        # 프로젝션
        svg_input_dim = element_embed_dim + command_embed_dim + num_continuous_params * 64
        self.svg_projection = nn.Linear(svg_input_dim, d_model)
        self.pixel_projection = nn.Linear(pixel_feature_dim, d_model)
        
        # 크로스 어텐션 (SVG ← Pixel)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_norm = nn.LayerNorm(d_model)
        
        # 트랜스포머 레이어
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        # 잠재 공간
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)
    
    def forward(self, svg_matrix, pixel_embedding, svg_mask=None):
        """
        svg_matrix: [B, L, 2+N] - element_id, command_id, params(양자화)
        pixel_embedding: [B, L, D_pixel]
        svg_mask: [B, L] - True=패딩
        """
        # SVG 임베딩
        elem_ids = svg_matrix[:, :, 0].long()
        cmd_ids = svg_matrix[:, :, 1].long()
        params = svg_matrix[:, :, 2:].long()
        
        elem_emb = self.element_embedding(elem_ids)
        cmd_emb = self.command_embedding(cmd_ids)
        
        # 파라미터 임베딩
        param_embs = []
        for i in range(params.size(-1)):
            param_embs.append(self.param_embedding(params[:, :, i]))
        
        # SVG 특징 결합
        svg_features = torch.cat([elem_emb, cmd_emb] + param_embs, dim=-1)
        svg_proj = self.svg_projection(svg_features)
        
        # 픽셀 프로젝션
        pixel_proj = self.pixel_projection(pixel_embedding)
        
        # 크로스 어텐션 (SVG가 픽셀 정보를 참조)
        cross_out = self.cross_attn(svg_proj, pixel_proj, pixel_proj, svg_mask)
        x = self.cross_norm(svg_proj + cross_out)
        
        # 트랜스포머
        for layer in self.layers:
            x = layer(x, svg_mask)
        
        # 잠재 벡터
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        # 패딩 위치 제로화
        if svg_mask is not None:
            valid_mask = (~svg_mask).unsqueeze(-1).float()
            mu = mu * valid_mask
            log_var = log_var * valid_mask
        
        return mu, log_var


class VPVAEDecoder(nn.Module):
    """VP-VAE 디코더 - 잠재 벡터에서 SVG 복원"""
    def __init__(self, num_element_types, num_command_types,
                 num_continuous_params, latent_dim, d_model,
                 num_layers, num_heads):
        super().__init__()
        
        self.fc_latent = nn.Linear(latent_dim, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # 출력 헤드
        self.element_head = nn.Linear(d_model, num_element_types)
        self.command_head = nn.Linear(d_model, num_command_types)
        self.param_heads = nn.ModuleList([
            nn.Linear(d_model, 256) for _ in range(num_continuous_params)
        ])
    
    def forward(self, z, target_len):
        """
        z: [B, L, latent_dim]
        target_len: 출력 시퀀스 길이
        """
        x = self.fc_latent(z)
        
        # 트랜스포머
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 출력
        element_logits = self.element_head(x)
        command_logits = self.command_head(x)
        param_logits = [head(x) for head in self.param_heads]
        
        return element_logits, command_logits, param_logits


class VPVAE(nn.Module):
    """Vector-Pixel VAE"""
    def __init__(self, num_element_types=7, num_command_types=14,
                 element_embed_dim=64, command_embed_dim=64,
                 num_continuous_params=12, pixel_feature_dim=384,
                 encoder_d_model=512, decoder_d_model=512,
                 encoder_layers=4, decoder_layers=4,
                 num_heads=8, latent_dim=128, max_seq_len=1024):
        super().__init__()
        
        self.encoder = VPVAEEncoder(
            num_element_types, num_command_types,
            element_embed_dim, command_embed_dim,
            num_continuous_params, pixel_feature_dim,
            encoder_d_model, encoder_layers, num_heads, latent_dim
        )
        
        self.decoder = VPVAEDecoder(
            num_element_types, num_command_types,
            num_continuous_params, latent_dim, decoder_d_model,
            decoder_layers, num_heads
        )
        
        self.max_seq_len = max_seq_len
    
    def reparameterize(self, mu, log_var):
        """재파라미터화 트릭"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, svg_matrix, pixel_embedding, svg_mask=None):
        """
        전방향 전파
        """
        # 인코딩
        mu, log_var = self.encoder(svg_matrix, pixel_embedding, svg_mask)
        
        # 재파라미터화
        z = self.reparameterize(mu, log_var)
        
        # 디코딩
        element_logits, command_logits, param_logits = self.decoder(
            z, target_len=self.max_seq_len
        )
        
        return element_logits, command_logits, param_logits, mu, log_var
