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
    """VP-VAE 디코더 - 잠재 벡터에서 연속 SVG 특징 복원 (tanh)"""
    def __init__(self, num_features, latent_dim, d_model,
                 num_layers, num_heads):
        super().__init__()

        self.fc_latent = nn.Linear(latent_dim, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 연속 출력 헤드: [B, L, num_features] with tanh → [-1, 1]
        self.output_projection = nn.Linear(d_model, num_features)

    def forward(self, z, target_len):
        """
        z: [B, L, latent_dim]
        target_len: 출력 시퀀스 길이

        Returns:
            predicted_continuous_svg_features: [B, L, F] in [-1, 1]
        """
        x = self.fc_latent(z)

        # 트랜스포머
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # 연속 출력 + tanh
        return torch.tanh(self.output_projection(x))


class VPVAE(nn.Module):
    """Vector-Pixel VAE"""
    def __init__(self, num_element_types=7, num_command_types=13,
                 element_embed_dim=64, command_embed_dim=64,
                 num_continuous_params=12, pixel_feature_dim=384,
                 encoder_d_model=512, decoder_d_model=512,
                 encoder_layers=4, decoder_layers=4,
                 num_heads=8, latent_dim=128, max_seq_len=1024):
        super().__init__()

        self.num_continuous_params = num_continuous_params
        self.num_element_types = num_element_types
        self.num_command_types = num_command_types
        self.latent_dim = latent_dim
        # 디코더 출력 특징 수: elem_id + cmd_id + continuous_params
        self.num_features = 2 + num_continuous_params

        self.encoder = VPVAEEncoder(
            num_element_types, num_command_types,
            element_embed_dim, command_embed_dim,
            num_continuous_params, pixel_feature_dim,
            encoder_d_model, encoder_layers, num_heads, latent_dim
        )

        self.decoder = VPVAEDecoder(
            num_features=self.num_features,
            latent_dim=latent_dim,
            d_model=decoder_d_model,
            num_layers=decoder_layers,
            num_heads=num_heads
        )

        self.max_seq_len = max_seq_len

    def reparameterize(self, mu, log_var):
        """재파라미터화 트릭"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl_loss(self, mu, log_var, mask=None):
        """
        KL divergence loss: D_KL(q(z|x) || N(0,I))
        수식: kl_div = -0.5 * sum(1 + logvar - mu^2 - exp(clamp(logvar, max=80)), dim=latent)
        """
        logvar_clamped = torch.clamp(log_var, max=80.0)
        # sum over latent dim → [B, L] (per-position KL) or [B] if no L
        kl_per_position = -0.5 * (1 + log_var - mu.pow(2) - logvar_clamped.exp()).sum(dim=-1)

        if mask is not None and kl_per_position.dim() > 1:
            # [B, L] with mask: average over valid positions
            valid_mask = (~mask).float()  # [B, L]
            kl_per_position = kl_per_position * valid_mask
            kl_loss = kl_per_position.sum() / (valid_mask.sum() + 1e-9)
        else:
            kl_loss = kl_per_position.mean()

        return kl_loss

    def _build_svg_matrix(self, svg_element_ids, svg_command_ids, svg_continuous_params):
        """입력을 svg_matrix 형태로 결합"""
        # [B, L, 2 + num_params]
        return torch.cat([
            svg_element_ids.unsqueeze(-1),
            svg_command_ids.unsqueeze(-1),
            svg_continuous_params
        ], dim=-1)

    def normalize_target(self, svg_element_ids, svg_command_ids, svg_continuous_params):
        """
        SVG 정수 행렬을 [-1, 1] 연속 타겟으로 정규화

        Args:
            svg_element_ids: [B, L] (0 ~ num_element_types-1)
            svg_command_ids: [B, L] (0 ~ num_command_types-1)
            svg_continuous_params: [B, L, 12] (0 ~ 255)

        Returns:
            target_continuous: [B, L, F] in [-1, 1]
        """
        elem_norm = svg_element_ids.float() / max(self.num_element_types - 1, 1) * 2 - 1
        cmd_norm = svg_command_ids.float() / max(self.num_command_types - 1, 1) * 2 - 1
        params_norm = svg_continuous_params.float() / 255.0 * 2 - 1

        return torch.cat([
            elem_norm.unsqueeze(-1),
            cmd_norm.unsqueeze(-1),
            params_norm
        ], dim=-1)  # [B, L, 2+12]

    def denormalize_output(self, continuous_features):
        """
        [-1, 1] 연속 출력을 정수 인덱스로 역정규화

        Args:
            continuous_features: [B, L, F] in [-1, 1]

        Returns:
            element_ids: [B, L]
            command_ids: [B, L]
            continuous_params: [B, L, 12]
        """
        elem_cont = continuous_features[:, :, 0]
        cmd_cont = continuous_features[:, :, 1]
        params_cont = continuous_features[:, :, 2:]

        element_ids = ((elem_cont + 1) / 2 * (self.num_element_types - 1)).round().clamp(0, self.num_element_types - 1).long()
        command_ids = ((cmd_cont + 1) / 2 * (self.num_command_types - 1)).round().clamp(0, self.num_command_types - 1).long()
        continuous_params = ((params_cont + 1) / 2 * 255).round().clamp(0, 255).long()

        return element_ids, command_ids, continuous_params

    def forward(self, svg_element_ids, svg_command_ids, svg_continuous_params,
                pixel_features, svg_mask=None):
        """
        전방향 전파

        Args:
            svg_element_ids: [B, L] - element type ids
            svg_command_ids: [B, L] - command type ids
            svg_continuous_params: [B, L, 12] - quantized continuous parameters
            pixel_features: [B, L, D_pixel] - DINOv2 pixel embeddings
            svg_mask: [B, L] - True=padding

        Returns:
            dict with 'predicted_features', 'mu', 'log_var', 'kl_loss'
        """
        # SVG matrix 구성 (인코더 입력)
        svg_matrix = self._build_svg_matrix(
            svg_element_ids, svg_command_ids, svg_continuous_params
        )

        # 인코딩
        mu, log_var = self.encoder(svg_matrix, pixel_features, svg_mask)

        # KL Loss 계산
        kl_loss = self.compute_kl_loss(mu, log_var, svg_mask)

        # 재파라미터화
        z = self.reparameterize(mu, log_var)

        # 디코딩 → 연속 특징 [B, L, F] in [-1, 1]
        predicted_features = self.decoder(z, target_len=z.size(1))

        return {
            'predicted_features': predicted_features,  # [B, L_out, F] in [-1,1]
            'mu': mu,                                  # [B, L, latent_dim]
            'log_var': log_var,                        # [B, L, latent_dim]
            'kl_loss': kl_loss                         # scalar
        }

    def encode(self, svg_element_ids, svg_command_ids, svg_continuous_params,
               pixel_features, svg_mask=None, deterministic=False):
        """
        인코딩만 수행 (DiT 훈련용 / latent 추출용)

        Args:
            deterministic: True이면 z=mu (샘플링 없음), False이면 reparameterize

        Returns:
            z: [B, L, latent_dim]
        """
        svg_matrix = self._build_svg_matrix(
            svg_element_ids, svg_command_ids, svg_continuous_params
        )

        mu, log_var = self.encoder(svg_matrix, pixel_features, svg_mask)

        if deterministic:
            return mu
        else:
            z = self.reparameterize(mu, log_var)
            return z

    def decode(self, z):
        """
        디코딩만 수행 (생성용)

        Args:
            z: [B, L, latent_dim]

        Returns:
            predicted_features: [B, L, F] in [-1, 1]
        """
        return self.decoder(z, target_len=z.size(1))
