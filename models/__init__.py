"""
SVG Fusion 모델 패키지
"""

from .vpvae import VPVAE
from .vsdit import VS_DiT

# Alias for backward compatibility
VSDiT = VS_DiT

__all__ = ['VPVAE', 'VS_DiT', 'VSDiT']
