"""
SVG 파싱 및 텐서 변환

svgparsing: SVGParser - SVG XML → 구조화 dict
svgtensor:  SVGToTensor - parsed dict → hybrid 시퀀스 텐서 [L, 2+P]
"""

import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

from config import (
    ELEMENT_TYPES, PATH_COMMAND_TYPES,
    NUM_ELEMENT_TYPES, NUM_COMMAND_TYPES, NUM_CONTINUOUS_PARAMS,
    N_BINS, DEFAULT_PARAM_VAL, BOS_IDX, EOS_IDX, PAD_IDX
)


class SVGParser:
    """SVG 파일 파서 (svgparsing)"""

    def __init__(self):
        self.supported_elements = ['path', 'circle', 'rect', 'ellipse']

    def parse_svg(self, svg_file_path: str) -> Dict:
        """
        SVG 파일 파싱

        Returns:
            {
                "viewport": {width, height, viewBox},
                "paths": [...],
                "circles": [...],
                "rects": [...],
                "ellipses": [...]
            }
        """
        try:
            tree = ET.parse(svg_file_path)
            root = tree.getroot()

            viewport = self._parse_viewport(root)

            grouped = {
                'paths': [],
                'circles': [],
                'rects': [],
                'ellipses': []
            }

            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                if tag in self.supported_elements:
                    parsed_elem = self._parse_element(elem, tag)
                    if parsed_elem:
                        etype = parsed_elem['type']
                        if etype == 'path':
                            grouped['paths'].append(parsed_elem)
                        elif etype == 'circle':
                            grouped['circles'].append(parsed_elem)
                        elif etype == 'rect':
                            grouped['rects'].append(parsed_elem)
                        elif etype == 'ellipse':
                            grouped['ellipses'].append(parsed_elem)

            return {
                'viewport': viewport,
                **grouped
            }

        except Exception as e:
            raise ValueError(f"Failed to parse SVG: {e}")

    def _parse_viewport(self, root) -> Dict:
        """Viewport 정보 추출"""
        width = float(root.get('width', 512))
        height = float(root.get('height', 512))

        viewbox_str = root.get('viewBox', f'0 0 {width} {height}')
        vb_parts = [float(x) for x in viewbox_str.split()]

        return {
            'width': width,
            'height': height,
            'viewBox': {
                'x': vb_parts[0],
                'y': vb_parts[1],
                'width': vb_parts[2],
                'height': vb_parts[3]
            }
        }

    def _parse_element(self, elem, tag: str) -> Optional[Dict]:
        """개별 element 파싱"""
        try:
            if tag == 'path':
                return self._parse_path(elem)
            elif tag == 'circle':
                return self._parse_circle(elem)
            elif tag == 'rect':
                return self._parse_rect(elem)
            elif tag == 'ellipse':
                return self._parse_ellipse(elem)
            else:
                return None
        except:
            return None

    def _parse_path(self, elem) -> Dict:
        """Path element 파싱"""
        d = elem.get('d', '')
        commands = self._parse_path_data(d)
        style = self._parse_style(elem)

        return {
            'type': 'path',
            'commands': commands,
            'style': style
        }

    def _parse_path_data(self, d: str) -> List[Dict]:
        """Path d 속성 파싱"""
        commands = []
        pattern = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
        matches = re.findall(pattern, d)

        for cmd, params in matches:
            params = params.strip().replace(',', ' ')
            params = [float(x) for x in params.split() if x]

            commands.append({
                'command': cmd,
                'values': params
            })

        return commands

    def _parse_circle(self, elem) -> Dict:
        """Circle element 파싱"""
        cx = float(elem.get('cx', 0))
        cy = float(elem.get('cy', 0))
        r = float(elem.get('r', 0))
        style = self._parse_style(elem)

        return {
            'type': 'circle',
            'cx': cx, 'cy': cy, 'r': r,
            'style': style
        }

    def _parse_rect(self, elem) -> Dict:
        """Rect element 파싱"""
        x = float(elem.get('x', 0))
        y = float(elem.get('y', 0))
        width = float(elem.get('width', 0))
        height = float(elem.get('height', 0))
        style = self._parse_style(elem)

        return {
            'type': 'rect',
            'x': x, 'y': y, 'width': width, 'height': height,
            'style': style
        }

    def _parse_ellipse(self, elem) -> Dict:
        """Ellipse element 파싱"""
        cx = float(elem.get('cx', 0))
        cy = float(elem.get('cy', 0))
        rx = float(elem.get('rx', 0))
        ry = float(elem.get('ry', 0))
        style = self._parse_style(elem)

        return {
            'type': 'ellipse',
            'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry,
            'style': style
        }

    def _parse_style(self, elem) -> Dict:
        """스타일 속성 파싱"""
        style = {
            'fill': elem.get('fill', '#000000'),
            'stroke': elem.get('stroke', 'none'),
            'stroke-width': elem.get('stroke-width', '1'),
            'opacity': elem.get('opacity', '1.0')
        }

        style_str = elem.get('style', '')
        if style_str:
            for item in style_str.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    style[key.strip()] = value.strip()

        return style

    @staticmethod
    def parse_color(color_str: str) -> Tuple[int, int, int]:
        """색상 문자열을 RGB로 변환"""
        color_str = color_str.strip().lower()

        if color_str.startswith('#'):
            hex_color = color_str[1:]
            if len(hex_color) == 3:
                hex_color = ''.join([c * 2 for c in hex_color])

            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)

        elif color_str.startswith('rgb'):
            nums = re.findall(r'\d+', color_str)
            return tuple(map(int, nums[:3]))

        else:
            return (0, 0, 0)


class SVGToTensor:
    """
    SVG를 hybrid 텐서로 변환 (svgtensor / SVGToTensor_Normalized)

    하이브리드 토큰 1행: [element_rho_idx, command_tau_idx, param_bins[0..P-1]]
    최종 텐서: [L, 2+P] (dtype: int64)
    """

    def __init__(self, max_seq_len=1024):
        self.max_seq_len = max_seq_len

        # Vocab (config에서 가져옴)
        self.ELEMENT_TYPES = ELEMENT_TYPES
        self.PATH_COMMAND_TYPES = PATH_COMMAND_TYPES

        # 역방향 매핑
        self.ELEMENT_TYPES_REV = {v: k for k, v in self.ELEMENT_TYPES.items()}
        self.PATH_COMMAND_TYPES_REV = {v: k for k, v in self.PATH_COMMAND_TYPES.items()}

        # 정규화 범위
        self.COORD_MIN, self.COORD_MAX = 0.0, 512.0
        self.COLOR_MIN, self.COLOR_MAX = 0, 255

        self.num_bins = N_BINS
        self.num_geom_params = 8
        self.num_fill_style_params = 4
        self.num_continuous_params = NUM_CONTINUOUS_PARAMS

        self.DEFAULT_PARAM_VAL = DEFAULT_PARAM_VAL

        self.parser = SVGParser()

    def quantize(self, value: float, min_val: float, max_val: float) -> int:
        """연속 값을 bin으로 양자화"""
        normalized = np.clip((value - min_val) / (max_val - min_val + 1e-8), 0, 1)
        bin_idx = int(normalized * (self.num_bins - 1))
        return np.clip(bin_idx, 0, self.num_bins - 1)

    def dequantize(self, bin_idx: int, min_val: float, max_val: float) -> float:
        """Bin 중심값으로 역양자화"""
        return (bin_idx + 0.5) / self.num_bins * (max_val - min_val) + min_val

    def svg_to_tensor(self, svg_file_path: str) -> torch.Tensor:
        """
        SVG 파일을 hybrid 텐서로 변환

        Returns:
            torch.Tensor: [L, 2+P] - BOS + content + EOS (+ optional PAD)
        """
        parsed = self.parser.parse_svg(svg_file_path)

        # 모든 요소 합치기
        all_elements = (
            parsed.get('paths', []) +
            parsed.get('rects', []) +
            parsed.get('circles', []) +
            parsed.get('ellipses', [])
        )

        tensor_rows = []

        for element in all_elements:
            elem_type = element['type']
            elem_id = self.ELEMENT_TYPES.get(elem_type, 0)

            style = element.get('style', {})
            style_params = self._extract_style_params(style)

            if elem_type == 'path':
                for cmd_data in element['commands']:
                    cmd = cmd_data['command']
                    # 대문자 → 소문자 변환 (스펙: 소문자만 사용)
                    cmd_lower = cmd.lower()
                    cmd_id = self.PATH_COMMAND_TYPES.get(cmd_lower, 0)
                    values = cmd_data['values']

                    geom_params = [self.DEFAULT_PARAM_VAL] * self.num_geom_params
                    for i, val in enumerate(values[:self.num_geom_params]):
                        geom_params[i] = self.quantize(val, self.COORD_MIN, self.COORD_MAX)

                    row = [elem_id, cmd_id] + geom_params + style_params
                    tensor_rows.append(row)

            elif elem_type in ['circle', 'rect', 'ellipse']:
                geom_params = self._extract_geom_params(element)
                cmd_id = self.PATH_COMMAND_TYPES['NO_CMD']
                row = [elem_id, cmd_id] + geom_params + style_params
                tensor_rows.append(row)

        if not tensor_rows:
            tensor_rows = [[self.DEFAULT_PARAM_VAL] * (2 + self.num_continuous_params)]

        content = torch.tensor(tensor_rows, dtype=torch.long)

        # BOS row 삽입
        num_cols = 2 + self.num_continuous_params
        bos_row = torch.zeros(1, num_cols, dtype=torch.long)
        bos_row[0, 0] = BOS_IDX

        # EOS row 삽입
        eos_row = torch.zeros(1, num_cols, dtype=torch.long)
        eos_row[0, 0] = EOS_IDX

        tensor = torch.cat([bos_row, content, eos_row], dim=0)

        # Truncation: max_seq_len 초과 시 자르고 마지막을 EOS로 강제
        if tensor.shape[0] > self.max_seq_len:
            tensor = tensor[:self.max_seq_len]
            tensor[-1, :] = 0
            tensor[-1, 0] = EOS_IDX

        return tensor

    def _extract_geom_params(self, element: Dict) -> List[int]:
        """기하학적 파라미터 추출 (양자화)"""
        params = [0.0] * 8

        if element['type'] == 'circle':
            params[0] = element.get('cx', 0)
            params[1] = element.get('cy', 0)
            params[2] = element.get('r', 0)

        elif element['type'] == 'rect':
            params[0] = element.get('x', 0)
            params[1] = element.get('y', 0)
            params[2] = element.get('width', 0)
            params[3] = element.get('height', 0)

        elif element['type'] == 'ellipse':
            params[0] = element.get('cx', 0)
            params[1] = element.get('cy', 0)
            params[2] = element.get('rx', 0)
            params[3] = element.get('ry', 0)

        return [self.quantize(p, self.COORD_MIN, self.COORD_MAX) for p in params]

    def _extract_style_params(self, style: Dict) -> List[int]:
        """스타일 파라미터 추출 (양자화)"""
        fill = style.get('fill', '#000000')

        if fill == 'none' or not fill:
            r, g, b = 128, 128, 128
        else:
            try:
                r, g, b = SVGParser.parse_color(fill)
            except:
                r, g, b = 128, 128, 128

        opacity = float(style.get('opacity', 1.0))

        return [
            self.quantize(r, self.COLOR_MIN, self.COLOR_MAX),
            self.quantize(g, self.COLOR_MIN, self.COLOR_MAX),
            self.quantize(b, self.COLOR_MIN, self.COLOR_MAX),
            self.quantize(opacity, 0.0, 1.0)
        ]
