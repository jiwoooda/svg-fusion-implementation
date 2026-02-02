"""
TensorToSVGHybrid (tensorsvg)

hybrid tensor [L, 2+P] (int bins) → SVG string/file로 복원

핵심: element/command 디코딩, bin→연속값 (bin center), path 그룹핑(m/z 경계), style 적용
"""

import torch
from typing import Optional

from config import (
    ELEMENT_TYPES, PATH_COMMAND_TYPES,
    N_BINS, NUM_CONTINUOUS_PARAMS, EOS_IDX, PAD_IDX, BOS_IDX
)


class TensorToSVGHybrid:
    """
    hybrid tensor → SVG 복원

    - 새 vocab 기반 역매핑
    - bin center dequantize: (bin_idx + 0.5) / N_BINS * (max - min) + min
    - actual_len 파라미터 지원
    - 소문자 명령어 처리, STYLE/DEF 스킵
    - path 그룹핑 (m/z 경계)
    """

    def __init__(self, num_bins=N_BINS,
                 coord_range=(0.0, 512.0),
                 color_range=(0, 255)):
        self.num_bins = num_bins
        self.coord_min, self.coord_max = coord_range
        self.color_min, self.color_max = color_range

        self.num_geom_params = 8
        self.num_style_params = 4

        # 역방향 매핑
        self.ELEMENT_TYPES_REV = {v: k for k, v in ELEMENT_TYPES.items()}
        self.CMD_TYPES_REV = {v: k for k, v in PATH_COMMAND_TYPES.items()}

    def dequantize_bin_center(self, bin_idx: int, min_val: float, max_val: float) -> float:
        """Bin 중심값 기준 역양자화"""
        return (bin_idx + 0.5) / self.num_bins * (max_val - min_val) + min_val

    def _decode_geom_params(self, row: torch.Tensor):
        """row[2:10] → 8개 기하학적 연속값"""
        return [
            self.dequantize_bin_center(int(row[2 + i].item()), self.coord_min, self.coord_max)
            for i in range(self.num_geom_params)
        ]

    def _decode_style(self, row: torch.Tensor) -> str:
        """row[10:14] → fill:rgb(...);opacity:... 스타일 문자열"""
        r = self.dequantize_bin_center(int(row[10].item()), self.color_min, self.color_max)
        g = self.dequantize_bin_center(int(row[11].item()), self.color_min, self.color_max)
        b = self.dequantize_bin_center(int(row[12].item()), self.color_min, self.color_max)
        a = self.dequantize_bin_center(int(row[13].item()), 0.0, 1.0)
        return f"fill:rgb({int(r)},{int(g)},{int(b)});opacity:{a:.2f}"

    def tensor_to_svg_string(self, tensor: torch.Tensor,
                              width: int = 512, height: int = 512,
                              actual_len: Optional[int] = None) -> str:
        """
        hybrid tensor [L, 2+P] → SVG XML 문자열

        Args:
            tensor: [L, 2+P] int64 텐서
            width, height: SVG 캔버스 크기
            actual_len: content 길이 (EOS/PAD 이전까지). None이면 자동 탐색.
        """
        if actual_len is None:
            actual_len = self._find_actual_len(tensor)

        svg_elements = []
        current_path_cmds = []
        current_style = "fill:rgb(0,0,0);opacity:1.00"

        for i in range(actual_len):
            row = tensor[i]
            elem_id = int(row[0].item())
            cmd_id = int(row[1].item())

            elem_type = self.ELEMENT_TYPES_REV.get(elem_id, '<PAD>')

            # 특수 토큰 스킵
            if elem_type in ('<BOS>', '<EOS>', '<PAD>'):
                continue

            geom = self._decode_geom_params(row)
            current_style = self._decode_style(row)
            cmd_type = self.CMD_TYPES_REV.get(cmd_id, 'NO_CMD')

            # STYLE/DEF 토큰 스킵
            if cmd_type in ('STYLE', 'DEF'):
                continue

            if elem_type == 'path' and cmd_type != 'NO_CMD':
                cmd_str = self._build_path_cmd(cmd_type, geom)

                if cmd_type == 'z':
                    # z 명령: 현재까지 모은 path 출력
                    if current_path_cmds:
                        path_d = " ".join(current_path_cmds + ["z"])
                        svg_elements.append(f'<path d="{path_d}" style="{current_style}"/>')
                        current_path_cmds = []
                elif cmd_str:
                    current_path_cmds.append(cmd_str)

            elif elem_type == 'circle':
                cx, cy, r = geom[0], geom[1], geom[2]
                svg_elements.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" style="{current_style}"/>'
                )

            elif elem_type == 'rect':
                x, y, w, h = geom[0], geom[1], geom[2], geom[3]
                svg_elements.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                    f'style="{current_style}"/>'
                )

            elif elem_type == 'ellipse':
                cx, cy, rx, ry = geom[0], geom[1], geom[2], geom[3]
                svg_elements.append(
                    f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
                    f'style="{current_style}"/>'
                )

        # 미완성 path flush
        if current_path_cmds:
            path_d = " ".join(current_path_cmds)
            svg_elements.append(f'<path d="{path_d}" style="{current_style}"/>')

        indent_elements = "\n  ".join(f"  {elem}" for elem in svg_elements)
        svg_content = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg width="{width}" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n'
            f'  {indent_elements}\n'
            f'</svg>'
        )

        return svg_content

    def tensor_to_svg_file(self, tensor: torch.Tensor,
                           output_file: str,
                           width: int = 512, height: int = 512,
                           actual_len: Optional[int] = None) -> str:
        """hybrid tensor → SVG 파일 저장"""
        svg_content = self.tensor_to_svg_string(tensor, width, height, actual_len)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        return svg_content

    def _build_path_cmd(self, cmd_type: str, geom: list) -> Optional[str]:
        """path 명령어 문자열 생성"""
        if cmd_type in ('m', 'l'):
            return f"{cmd_type} {geom[0]:.2f},{geom[1]:.2f}"
        elif cmd_type == 'h':
            return f"h {geom[0]:.2f}"
        elif cmd_type == 'v':
            return f"v {geom[0]:.2f}"
        elif cmd_type == 'c':
            return (f"c {geom[0]:.2f},{geom[1]:.2f} "
                    f"{geom[2]:.2f},{geom[3]:.2f} "
                    f"{geom[4]:.2f},{geom[5]:.2f}")
        elif cmd_type == 's':
            return (f"s {geom[0]:.2f},{geom[1]:.2f} "
                    f"{geom[2]:.2f},{geom[3]:.2f}")
        elif cmd_type == 'q':
            return (f"q {geom[0]:.2f},{geom[1]:.2f} "
                    f"{geom[2]:.2f},{geom[3]:.2f}")
        elif cmd_type == 't':
            return f"t {geom[0]:.2f},{geom[1]:.2f}"
        elif cmd_type == 'a':
            return (f"a {geom[0]:.2f},{geom[1]:.2f} "
                    f"{geom[2]:.2f} {int(geom[3])} {int(geom[4])} "
                    f"{geom[5]:.2f},{geom[6]:.2f}")
        elif cmd_type == 'z':
            return None  # z는 별도 처리
        else:
            return None

    def _find_actual_len(self, tensor: torch.Tensor) -> int:
        """텐서에서 첫 EOS 또는 PAD 위치 찾기"""
        elem_ids = tensor[:, 0]
        for i in range(len(elem_ids)):
            eid = int(elem_ids[i].item())
            if eid in (EOS_IDX, PAD_IDX):
                return i
        return len(elem_ids)


def tensor_to_svg_file_hybrid_wrapper(tensor: torch.Tensor,
                                       output_file: str,
                                       actual_len: Optional[int] = None,
                                       width: int = 512,
                                       height: int = 512) -> str:
    """편의 함수: tensor → SVG 파일"""
    converter = TensorToSVGHybrid()
    return converter.tensor_to_svg_file(tensor, output_file, width, height, actual_len)
