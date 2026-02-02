"""
Progressive Accumulation Dataset (v5)

SVG 요소를 누적(progressive)으로 추가하면서:
1. 누적 SVG 래스터화 → DINOv2 patch embedding 시퀀스 생성
2. 누적 요소까지의 SVG hybrid 시퀀스 저장

Dataset은 step별 샘플을 동적으로 로드하여:
- svg_query: BOS + content + EOS + PAD
- pixel_kv_seq: CLS repeat + pad (또는 patch embeddings)
- attention_mask
반환
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import cairosvg
from io import BytesIO
from transformers import AutoImageProcessor, AutoModel
import xml.etree.ElementTree as ET

from utils.svg_parser import SVGParser, SVGToTensor
from utils.hybrid_utils import build_special_rows, pad_or_truncate
from config import (
    PAD_IDX, BOS_IDX, EOS_IDX,
    NUM_CONTINUOUS_PARAMS, N_BINS
)


class ProgressiveAccumulationDataset(Dataset):
    """
    Progressive 누적 데이터셋

    SVG에 N개 요소가 있으면 N개 훈련 샘플 생성:
    - step 1: 첫 번째 요소만 포함
    - step 2: 첫 + 두 번째 요소
    - ...
    - step N: 모든 요소 포함
    """

    def __init__(self, svg_dir, max_seq_len=1024,
                 dino_model_name='facebook/dinov2-small',
                 image_size=224, use_patch_embeddings=True,
                 captions_dict=None):
        """
        Args:
            svg_dir: SVG 파일 디렉토리
            max_seq_len: 최대 시퀀스 길이
            dino_model_name: DINOv2 모델
            image_size: 래스터화 크기
            use_patch_embeddings: True이면 patch tokens, False이면 CLS token
            captions_dict: {filename: caption}
        """
        self.svg_dir = Path(svg_dir)
        self.svg_files = sorted(list(self.svg_dir.glob('**/*.svg')))
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.use_patch_embeddings = use_patch_embeddings
        self.captions_dict = captions_dict or {}

        if not self.svg_files:
            print(f"Warning: No SVG files found in {svg_dir}")

        # SVG 변환기
        self.converter = SVGToTensor(max_seq_len=max_seq_len)
        self.parser = SVGParser()

        # Special rows
        self.bos_row, self.eos_row, self.pad_row = build_special_rows(self.converter)

        # DINOv2
        print(f"Loading DINOv2: {dino_model_name}")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModel.from_pretrained(dino_model_name)
        self.dino_model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dino_model.to(self.device)
        self.dino_embed_dim = self.dino_model.config.hidden_size

        # 인덱스 빌드: (svg_file_idx, num_accumulated_elements)
        self.samples = self._build_sample_index()
        print(f"Progressive dataset: {len(self.svg_files)} SVGs, {len(self.samples)} total samples")

    def _build_sample_index(self):
        """각 SVG에서 progressive step별 샘플 인덱스 구성"""
        samples = []
        for file_idx, svg_path in enumerate(self.svg_files):
            try:
                parsed = self.parser.parse_svg(str(svg_path))
                all_elements = (
                    parsed.get('paths', []) +
                    parsed.get('rects', []) +
                    parsed.get('circles', []) +
                    parsed.get('ellipses', [])
                )
                num_elements = len(all_elements)
                for step in range(1, num_elements + 1):
                    samples.append((file_idx, step))
            except Exception as e:
                print(f"Skipping {svg_path}: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def _render_partial_svg(self, svg_path, num_elements):
        """SVG에서 처음 num_elements개 요소만 포함한 이미지 생성"""
        try:
            tree = ET.parse(str(svg_path))
            root = tree.getroot()
            ns = ''
            if '}' in root.tag:
                ns = root.tag.split('}')[0] + '}'

            # 지원 요소만 필터
            supported = {'path', 'circle', 'rect', 'ellipse'}
            svg_elements = []
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag in supported:
                    svg_elements.append(elem)

            # num_elements개만 유지, 나머지 제거
            for elem in svg_elements[num_elements:]:
                parent = root.find('.//' + elem.tag + '/..', {})
                if parent is None:
                    # root 직접 자식
                    root.remove(elem)
                else:
                    parent.remove(elem)

            # SVG 문자열로 변환
            svg_str = ET.tostring(root, encoding='unicode')

            # 래스터화
            png_buffer = BytesIO()
            cairosvg.svg2png(
                bytestring=svg_str.encode('utf-8'),
                write_to=png_buffer,
                output_width=self.image_size,
                output_height=self.image_size
            )
            png_buffer.seek(0)
            return Image.open(png_buffer).convert('RGB')

        except Exception:
            return Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))

    def _get_dino_embedding(self, image):
        """DINOv2 임베딩 추출"""
        with torch.no_grad():
            inputs = self.dino_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.dino_model(**inputs)

            if self.use_patch_embeddings:
                # patch tokens (CLS 제외): [1, num_patches, D]
                embedding = outputs.last_hidden_state[:, 1:].cpu()
            else:
                # CLS token: [1, D]
                embedding = outputs.last_hidden_state[:, 0:1].cpu()

        return embedding.squeeze(0)  # [num_patches, D] or [1, D]

    def _build_partial_tensor(self, svg_path, num_elements):
        """처음 num_elements개 요소의 hybrid tensor 생성 (BOS/EOS 포함)"""
        parsed = self.parser.parse_svg(str(svg_path))
        all_elements = (
            parsed.get('paths', []) +
            parsed.get('rects', []) +
            parsed.get('circles', []) +
            parsed.get('ellipses', [])
        )

        elements = all_elements[:num_elements]
        tensor_rows = []

        for element in elements:
            elem_type = element['type']
            elem_id = self.converter.ELEMENT_TYPES.get(elem_type, 0)
            style = element.get('style', {})
            style_params = self.converter._extract_style_params(style)

            if elem_type == 'path':
                for cmd_data in element['commands']:
                    cmd = cmd_data['command']
                    cmd_lower = cmd.lower()
                    cmd_id = self.converter.PATH_COMMAND_TYPES.get(cmd_lower, 0)
                    values = cmd_data['values']

                    geom_params = [0] * self.converter.num_geom_params
                    for i, val in enumerate(values[:self.converter.num_geom_params]):
                        geom_params[i] = self.converter.quantize(
                            val, self.converter.COORD_MIN, self.converter.COORD_MAX
                        )

                    row = [elem_id, cmd_id] + geom_params + style_params
                    tensor_rows.append(row)

            elif elem_type in ['circle', 'rect', 'ellipse']:
                geom_params = self.converter._extract_geom_params(element)
                cmd_id = self.converter.PATH_COMMAND_TYPES['NO_CMD']
                row = [elem_id, cmd_id] + geom_params + style_params
                tensor_rows.append(row)

        if not tensor_rows:
            tensor_rows = [[0] * (2 + self.converter.num_continuous_params)]

        content = torch.tensor(tensor_rows, dtype=torch.long)

        # BOS + content + EOS
        num_cols = 2 + self.converter.num_continuous_params
        bos = self.bos_row.unsqueeze(0)
        eos = self.eos_row.unsqueeze(0)
        seq = torch.cat([bos, content, eos], dim=0)

        # pad_or_truncate
        seq = pad_or_truncate(seq, self.max_seq_len,
                              self.bos_row, self.eos_row, self.pad_row)

        return seq

    def __getitem__(self, idx):
        file_idx, num_elements = self.samples[idx]
        svg_path = self.svg_files[file_idx]

        # 1. 누적 SVG hybrid tensor (BOS + content + EOS + PAD)
        svg_query = self._build_partial_tensor(svg_path, num_elements)

        # 2. 누적 SVG 래스터화 → DINOv2 embedding
        image = self._render_partial_svg(svg_path, num_elements)
        pixel_emb = self._get_dino_embedding(image)  # [num_patches, D] or [1, D]

        # 3. pixel_kv_seq: svg_query 길이에 맞춰 반복/패딩
        seq_len = self.max_seq_len
        if self.use_patch_embeddings:
            # patch embeddings: [num_patches, D] → [seq_len, D]
            num_patches = pixel_emb.shape[0]
            if num_patches >= seq_len:
                pixel_kv_seq = pixel_emb[:seq_len]
            else:
                repeat_count = (seq_len // num_patches) + 1
                pixel_kv_seq = pixel_emb.repeat(repeat_count, 1)[:seq_len]
        else:
            # CLS repeat: [1, D] → [seq_len, D]
            pixel_kv_seq = pixel_emb.repeat(seq_len, 1)

        # 4. attention_mask (True = padding)
        attention_mask = torch.zeros(seq_len, dtype=torch.bool)
        elem_ids = svg_query[:, 0]
        for i in range(seq_len):
            if int(elem_ids[i].item()) == PAD_IDX:
                attention_mask[i:] = True
                break

        # 캡션
        filename_stem = svg_path.stem
        caption = self.captions_dict.get(filename_stem, filename_stem)

        return {
            'svg_query': svg_query,              # [max_seq_len, 2+P]
            'pixel_kv_seq': pixel_kv_seq,        # [max_seq_len, D]
            'attention_mask': attention_mask,     # [max_seq_len]
            'caption': caption,
            'filename': filename_stem,
            'num_elements': num_elements
        }


def progressive_collate_fn(batch):
    """Progressive 데이터셋용 collate"""
    return {
        'svg_query': torch.stack([item['svg_query'] for item in batch]),
        'pixel_kv_seq': torch.stack([item['pixel_kv_seq'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'caption': [item['caption'] for item in batch],
        'filename': [item['filename'] for item in batch],
        'num_elements': [item['num_elements'] for item in batch]
    }
