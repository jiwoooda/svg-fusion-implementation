"""
SVG 파싱 및 텐서 변환
"""

import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


class SVGParser:
    """SVG 파일 파서"""
    
    def __init__(self):
        self.supported_elements = ['path', 'circle', 'rect', 'ellipse', 'line', 'polyline', 'polygon']
    
    def parse_svg(self, svg_file_path: str) -> Dict:
        """SVG 파일 파싱"""
        try:
            tree = ET.parse(svg_file_path)
            root = tree.getroot()
            
            viewport = self._parse_viewport(root)
            
            elements = []
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                if tag in self.supported_elements:
                    parsed_elem = self._parse_element(elem, tag)
                    if parsed_elem:
                        elements.append(parsed_elem)
            
            return {
                'viewport': viewport,
                'elements': elements
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
            'commands': [
                {'command': 'cx', 'values': [cx]},
                {'command': 'cy', 'values': [cy]},
                {'command': 'r', 'values': [r]}
            ],
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
            'commands': [
                {'command': 'x', 'values': [x]},
                {'command': 'y', 'values': [y]},
                {'command': 'width', 'values': [width]},
                {'command': 'height', 'values': [height]}
            ],
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
            'commands': [
                {'command': 'cx', 'values': [cx]},
                {'command': 'cy', 'values': [cy]},
                {'command': 'rx', 'values': [rx]},
                {'command': 'ry', 'values': [ry]}
            ],
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
                hex_color = ''.join([c*2 for c in hex_color])
            
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
    """SVG를 텐서로 변환"""
    
    def __init__(self, max_seq_len=1024):
        self.max_seq_len = max_seq_len
        
        # Element types
        self.ELEMENT_TYPES = {
            '<PAD>': 0, '<BOS>': 1, '<EOS>': 2,
            'path': 3, 'circle': 4, 'rect': 5, 'ellipse': 6
        }
        
        # Command types
        self.PATH_COMMAND_TYPES = {
            'NO_CMD': 0, 'M': 1, 'm': 2, 'L': 3, 'l': 4,
            'H': 5, 'h': 6, 'V': 7, 'v': 8, 'C': 9, 'c': 10,
            'Z': 11, 'z': 12, 'cx': 13
        }
        
        # 역방향 매핑
        self.ELEMENT_TYPES_REV = {v: k for k, v in self.ELEMENT_TYPES.items()}
        self.PATH_COMMAND_TYPES_REV = {v: k for k, v in self.PATH_COMMAND_TYPES.items()}
        
        # 정규화 범위
        self.COORD_MIN, self.COORD_MAX = 0.0, 512.0
        self.COLOR_MIN, self.COLOR_MAX = 0, 255
        
        self.num_bins = 256
        self.num_geom_params = 8
        self.num_fill_style_params = 4
        self.num_continuous_params = self.num_geom_params + self.num_fill_style_params
        
        self.DEFAULT_PARAM_VAL = 0.0
        
        self.parser = SVGParser()
    
    def quantize(self, value: float, min_val: float, max_val: float) -> int:
        """연속 값을 bin으로 양자화"""
        normalized = np.clip((value - min_val) / (max_val - min_val + 1e-8), 0, 1)
        bin_idx = int(normalized * (self.num_bins - 1))
        return np.clip(bin_idx, 0, self.num_bins - 1)
    
    def dequantize(self, bin_idx: int, min_val: float, max_val: float) -> float:
        """Bin을 연속 값으로 역양자화"""
        normalized = bin_idx / (self.num_bins - 1)
        return normalized * (max_val - min_val) + min_val
    
    def svg_to_tensor(self, svg_file_path: str) -> torch.Tensor:
        """
        SVG 파일을 텐서로 변환
        
        Returns:
            torch.Tensor: [L, 2+12] - (elem_id, cmd_id, 8 geom, 4 style)
        """
        parsed = self.parser.parse_svg(svg_file_path)
        
        tensor_rows = []
        
        for element in parsed['elements']:
            elem_type = element['type']
            elem_id = self.ELEMENT_TYPES.get(elem_type, 0)
            
            style = element.get('style', {})
            style_params = self._extract_style_params(style)
            
            if elem_type == 'path':
                for cmd_data in element['commands']:
                    cmd = cmd_data['command']
                    cmd_id = self.PATH_COMMAND_TYPES.get(cmd, 0)
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
            return torch.zeros((1, 2 + self.num_continuous_params), dtype=torch.long)
        
        tensor = torch.tensor(tensor_rows, dtype=torch.long)
        
        if tensor.shape[0] > self.max_seq_len:
            tensor = tensor[:self.max_seq_len]
        
        return tensor
    
    def _extract_geom_params(self, element: Dict) -> List[int]:
        """기하학적 파라미터 추출 (양자화)"""
        commands = element['commands']
        params = [0.0] * 8
        
        if element['type'] == 'circle':
            params[0] = commands[0]['values'][0]  # cx
            params[1] = commands[1]['values'][0]  # cy
            params[2] = commands[2]['values'][0]  # r
        
        elif element['type'] == 'rect':
            params[0] = commands[0]['values'][0]  # x
            params[1] = commands[1]['values'][0]  # y
            params[2] = commands[2]['values'][0]  # width
            params[3] = commands[3]['values'][0]  # height
        
        elif element['type'] == 'ellipse':
            params[0] = commands[0]['values'][0]  # cx
            params[1] = commands[1]['values'][0]  # cy
            params[2] = commands[2]['values'][0]  # rx
            params[3] = commands[3]['values'][0]  # ry
        
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
    
    def tensor_to_svg(self, tensor: torch.Tensor, output_file: str, 
                     width: int = 512, height: int = 512) -> str:
        """텐서를 SVG 파일로 변환"""
        svg_elements = []
        current_path_cmds = []
        current_style = "fill:rgb(128,128,128);opacity:1.0"
        
        for i in range(tensor.shape[0]):
            row = tensor[i]
            elem_id = int(row[0].item())
            cmd_id = int(row[1].item())
            
            # 파라미터 역양자화
            geom_params = [
                self.dequantize(int(row[j].item()), self.COORD_MIN, self.COORD_MAX)
                for j in range(2, 2 + self.num_geom_params)
            ]
            
            style_r = self.dequantize(int(row[10].item()), self.COLOR_MIN, self.COLOR_MAX)
            style_g = self.dequantize(int(row[11].item()), self.COLOR_MIN, self.COLOR_MAX)
            style_b = self.dequantize(int(row[12].item()), self.COLOR_MIN, self.COLOR_MAX)
            style_a = self.dequantize(int(row[13].item()), 0.0, 1.0)
            
            current_style = f"fill:rgb({int(style_r)},{int(style_g)},{int(style_b)});opacity:{style_a:.2f}"
            
            elem_type = self.ELEMENT_TYPES_REV.get(elem_id, '<PAD>')
            
            if elem_type in ['<PAD>', '<EOS>']:
                break
            
            if elem_type == '<BOS>':
                continue
            
            cmd_type = self.PATH_COMMAND_TYPES_REV.get(cmd_id, 'NO_CMD')
            
            if elem_type == 'path' and cmd_type != 'NO_CMD':
                if cmd_type in ['M', 'm']:
                    cmd_str = f"{cmd_type} {geom_params[0]:.2f},{geom_params[1]:.2f}"
                elif cmd_type in ['L', 'l']:
                    cmd_str = f"{cmd_type} {geom_params[0]:.2f},{geom_params[1]:.2f}"
                elif cmd_type in ['C', 'c']:
                    cmd_str = (f"{cmd_type} {geom_params[0]:.2f},{geom_params[1]:.2f} "
                             f"{geom_params[2]:.2f},{geom_params[3]:.2f} "
                             f"{geom_params[4]:.2f},{geom_params[5]:.2f}")
                elif cmd_type in ['Z', 'z']:
                    cmd_str = cmd_type
                    if current_path_cmds:
                        path_d = " ".join(current_path_cmds + [cmd_str])
                        svg_elements.append(f'<path d="{path_d}" style="{current_style}"/>')
                        current_path_cmds = []
                    continue
                else:
                    cmd_str = None
                
                if cmd_str:
                    current_path_cmds.append(cmd_str)
            
            elif elem_type == 'circle':
                cx, cy, r = geom_params[0], geom_params[1], geom_params[2]
                svg_elements.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" style="{current_style}"/>'
                )
            
            elif elem_type == 'rect':
                x, y, w, h = geom_params[0], geom_params[1], geom_params[2], geom_params[3]
                svg_elements.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                    f'style="{current_style}"/>'
                )
            
            elif elem_type == 'ellipse':
                cx, cy, rx, ry = geom_params[0], geom_params[1], geom_params[2], geom_params[3]
                svg_elements.append(
                    f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
                    f'style="{current_style}"/>'
                )
        
        if current_path_cmds:
            path_d = " ".join(current_path_cmds)
            svg_elements.append(f'<path d="{path_d}" style="{current_style}"/>')
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  {chr(10).join(f"  {elem}" for elem in svg_elements)}
</svg>'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return svg_content
