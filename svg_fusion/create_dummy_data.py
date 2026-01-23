"""
테스트용 더미 SVG 파일 생성
"""

import os
from pathlib import Path
import argparse


def create_dummy_svgs(output_dir, num_samples=10):
    """테스트용 SVG 파일 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 다양한 SVG 샘플
    svgs = []
    
    # 1. 원
    for i in range(num_samples // 5):
        cx, cy = 100 + i * 50, 100 + i * 50
        r = 30 + i * 10
        color = ['red', 'blue', 'green', 'yellow', 'purple'][i % 5]
        svgs.append((f'circle_{i}.svg', f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="0.8"/>
</svg>'''))
    
    # 2. 사각형
    for i in range(num_samples // 5):
        x, y = 50 + i * 60, 50 + i * 60
        w, h = 80, 60
        color = ['#FF5733', '#33FF57', '#3357FF', '#FF33F5', '#F5FF33'][i % 5]
        svgs.append((f'rect_{i}.svg', f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" opacity="0.7"/>
</svg>'''))
    
    # 3. 타원
    for i in range(num_samples // 5):
        cx, cy = 150 + i * 40, 150 + i * 40
        rx, ry = 50 + i * 5, 30 + i * 5
        svgs.append((f'ellipse_{i}.svg', f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="orange" opacity="0.6"/>
</svg>'''))
    
    # 4. 간단한 경로
    for i in range(num_samples // 5):
        x1, y1 = 50, 50 + i * 40
        x2, y2 = 200, 50 + i * 40
        x3, y3 = 200, 200
        svgs.append((f'path_{i}.svg', f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <path d="M {x1} {y1} L {x2} {y2} L {x3} {y3} Z" fill="teal" opacity="0.8"/>
</svg>'''))
    
    # 5. 복합 도형
    for i in range(num_samples - len(svgs)):
        svgs.append((f'multi_{i}.svg', f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <circle cx="100" cy="100" r="40" fill="red"/>
  <rect x="200" y="200" width="100" height="80" fill="blue"/>
  <ellipse cx="400" cy="400" rx="60" ry="40" fill="green"/>
</svg>'''))
    
    # 파일 저장
    for filename, content in svgs:
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(svgs)} dummy SVG files in {output_dir}")
    return len(svgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dummy SVG files for testing")
    parser.add_argument('--output_dir', type=str, default='data/svgs', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of SVG files to create')
    
    args = parser.parse_args()
    
    create_dummy_svgs(args.output_dir, args.num_samples)
