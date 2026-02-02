"""
SVG Fusion 유틸리티 패키지
"""

from .svg_parser import SVGParser, SVGToTensor
from .tensorsvg import TensorToSVGHybrid, tensor_to_svg_file_hybrid_wrapper
from .hybrid_utils import (
    build_special_rows,
    pad_or_truncate,
    decode_model_outputs_to_hybrid,
    compute_actual_len,
    save_svg_from_hybrid
)
from .diffusion import DiffusionUtils

try:
    from .dataset import SVGDataset, collate_fn
except ImportError:
    SVGDataset = None
    collate_fn = None

__all__ = [
    'SVGParser', 'SVGToTensor',
    'TensorToSVGHybrid', 'tensor_to_svg_file_hybrid_wrapper',
    'build_special_rows', 'pad_or_truncate',
    'decode_model_outputs_to_hybrid', 'compute_actual_len', 'save_svg_from_hybrid',
    'DiffusionUtils', 'SVGDataset', 'collate_fn'
]
