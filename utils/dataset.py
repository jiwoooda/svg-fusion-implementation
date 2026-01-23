"""
완전한 SVG 데이터셋 - DINOv2 픽셀 임베딩 포함
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import cairosvg
from io import BytesIO
from transformers import AutoImageProcessor, AutoModel
from .svg_parser import SVGToTensor


class SVGDataset(Dataset):
    """실제 작동하는 SVG 데이터셋"""
    
    def __init__(self, svg_dir, captions_dict=None, max_seq_len=1024,
                 dino_model_name='facebook/dinov2-small', cache_embeddings=True,
                 image_size=224):
        """
        Args:
            svg_dir: SVG 파일이 있는 디렉토리
            captions_dict: {filename: caption} 딕셔너리
            max_seq_len: 최대 시퀀스 길이
            dino_model_name: DINOv2 모델 이름
            cache_embeddings: 픽셀 임베딩 캐시 여부
            image_size: 래스터화 이미지 크기
        """
        self.svg_dir = Path(svg_dir)
        self.svg_files = sorted(list(self.svg_dir.glob('**/*.svg')))
        self.captions_dict = captions_dict or {}
        self.max_seq_len = max_seq_len
        self.cache_embeddings = cache_embeddings
        self.image_size = image_size
        
        if not self.svg_files:
            print(f"Warning: No SVG files found in {svg_dir}")
        
        print(f"Found {len(self.svg_files)} SVG files")
        
        # SVG 변환기
        self.svg_converter = SVGToTensor(max_seq_len=max_seq_len)
        
        # DINOv2 설정
        print(f"Loading DINOv2: {dino_model_name}")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModel.from_pretrained(dino_model_name)
        self.dino_model.eval()
        
        # 디바이스
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dino_model.to(self.device)
        
        self.dino_embed_dim = self.dino_model.config.hidden_size
        
        # 캐시
        self._embedding_cache = {} if cache_embeddings else None
        
        print(f"Dataset initialized: {len(self)} samples, DINOv2 dim: {self.dino_embed_dim}")
    
    def __len__(self):
        return len(self.svg_files)
    
    def _svg_to_image(self, svg_path):
        """SVG를 PIL 이미지로 변환"""
        try:
            png_buffer = BytesIO()
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=png_buffer,
                output_width=self.image_size,
                output_height=self.image_size
            )
            png_buffer.seek(0)
            image = Image.open(png_buffer).convert('RGB')
            return image
        except Exception as e:
            print(f"Error converting {svg_path} to image: {e}")
            return Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
    
    def _get_pixel_embedding(self, svg_path):
        """DINOv2 픽셀 임베딩 추출"""
        # 캐시 확인
        if self._embedding_cache is not None:
            svg_key = str(svg_path)
            if svg_key in self._embedding_cache:
                return self._embedding_cache[svg_key]
        
        # 이미지 변환
        image = self._svg_to_image(svg_path)
        
        # DINOv2 임베딩
        with torch.no_grad():
            inputs = self.dino_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.dino_model(**inputs)
            # CLS 토큰 [1, hidden_dim]
            embedding = outputs.last_hidden_state[:, 0].cpu()
        
        # 캐시
        if self._embedding_cache is not None:
            self._embedding_cache[svg_key] = embedding
        
        return embedding
    
    def __getitem__(self, idx):
        svg_path = self.svg_files[idx]
        
        # SVG를 텐서로 변환
        try:
            svg_tensor = self.svg_converter.svg_to_tensor(str(svg_path))
        except Exception as e:
            print(f"Error processing {svg_path}: {e}")
            # 더미 텐서
            num_params = 2 + self.svg_converter.num_continuous_params
            svg_tensor = torch.zeros((1, num_params), dtype=torch.long)
        
        # 픽셀 임베딩
        pixel_embedding = self._get_pixel_embedding(svg_path)  # [1, D]
        
        # 시퀀스 길이
        seq_len = svg_tensor.shape[0]
        
        # 픽셀 임베딩을 시퀀스 길이만큼 반복 [L, D]
        pixel_embedding_seq = pixel_embedding.repeat(seq_len, 1)
        
        # 마스크 (False = valid, True = padding)
        svg_mask = torch.zeros(seq_len, dtype=torch.bool)
        
        # 캡션
        filename_stem = svg_path.stem
        caption = self.captions_dict.get(filename_stem, filename_stem)
        
        return {
            'svg_tensor': svg_tensor,
            'pixel_embedding': pixel_embedding_seq,
            'svg_mask': svg_mask,
            'caption': caption,
            'filename': filename_stem
        }


def collate_fn(batch):
    """배치 collate 함수"""
    # 최대 시퀀스 길이
    max_len = max(item['svg_tensor'].shape[0] for item in batch)
    
    batch_svg = []
    batch_pixel = []
    batch_mask = []
    batch_captions = []
    batch_filenames = []
    
    for item in batch:
        svg = item['svg_tensor']
        pixel = item['pixel_embedding']
        mask = item['svg_mask']
        
        # 패딩
        pad_len = max_len - svg.shape[0]
        
        if pad_len > 0:
            # SVG 텐서 패딩
            pad_row = torch.zeros((pad_len, svg.shape[1]), dtype=torch.long)
            svg_padded = torch.cat([svg, pad_row], dim=0)
            
            # 픽셀 임베딩 패딩
            pixel_pad = torch.zeros((pad_len, pixel.shape[1]), dtype=torch.float32)
            pixel_padded = torch.cat([pixel, pixel_pad], dim=0)
            
            # 마스크 (True = padding)
            mask_pad = torch.ones(pad_len, dtype=torch.bool)
            mask_padded = torch.cat([mask, mask_pad], dim=0)
        else:
            svg_padded = svg
            pixel_padded = pixel
            mask_padded = mask
        
        batch_svg.append(svg_padded)
        batch_pixel.append(pixel_padded)
        batch_mask.append(mask_padded)
        batch_captions.append(item['caption'])
        batch_filenames.append(item['filename'])
    
    return {
        'svg_tensor': torch.stack(batch_svg),
        'pixel_embedding': torch.stack(batch_pixel),
        'svg_mask': torch.stack(batch_mask),
        'caption': batch_captions,
        'filename': batch_filenames
    }
