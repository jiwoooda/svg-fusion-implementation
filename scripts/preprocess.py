import argparse
import os
import random
from pathlib import Path
from io import BytesIO
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
import cairosvg
from transformers import AutoImageProcessor, AutoModel

from utils.svg_parser import SVGToTensor
import unicodedata


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def has_unsupported_features(svg_path: Path) -> bool:
    banned_tags = {"defs", "use", "clippath", "mask", "filter"}
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception:
        return True

    has_supported = False

    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag in banned_tags:
            return True
        if tag in ("path", "circle", "rect", "ellipse"):
            has_supported = True
        if "transform" in elem.attrib:
            return True

    return not has_supported


def svg_to_image(svg_path: Path, image_size: int) -> Image.Image:
    png_buffer = BytesIO()
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=png_buffer,
        output_width=image_size,
        output_height=image_size,
    )
    png_buffer.seek(0)
    return Image.open(png_buffer).convert("RGB")


def caption_from_filename(stem: str) -> str:
    parts = stem.split("-")
    chars = []
    try:
        for p in parts:
            if not p:
                continue
            code = int(p, 16)
            chars.append(chr(code))
    except Exception:
        return stem

    if not chars:
        return stem

    names = []
    for ch in chars:
        try:
            names.append(unicodedata.name(ch).lower())
        except ValueError:
            names.append("emoji")
    return " ".join(names)


def main():
    parser = argparse.ArgumentParser(description="Preprocess SVGs into cached .pt files")
    parser.add_argument("--svg_dir", type=str, required=True, help="SVG root directory")
    parser.add_argument("--pattern", type=str, default="*.svg", help="Glob pattern")
    parser.add_argument("--precomputed_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--max_files", type=int, default=300, help="Limit number of SVGs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument("--dino_model", type=str, default="facebook/dinov2-small", help="DINOv2 model")
    parser.add_argument("--image_size", type=int, default=224, help="Raster size for DINOv2")
    args = parser.parse_args()

    set_seed(args.seed)

    svg_dir = Path(args.svg_dir)
    if not svg_dir.exists():
        raise FileNotFoundError(f"svg_dir not found: {svg_dir}")

    precomputed_dir = Path(args.precomputed_dir)
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(svg_dir.rglob(args.pattern))
    if args.max_files > 0:
        svg_files = svg_files[: args.max_files]

    if not svg_files:
        print("No SVG files found.")
        return

    print(f"Found {len(svg_files)} SVG files (max_files={args.max_files})")
    print("Loading DINOv2...")

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.dino_model)
    model = AutoModel.from_pretrained(args.dino_model).to(device)
    model.eval()

    svg_converter = SVGToTensor(max_seq_len=1024)

    saved = 0
    skipped = 0
    for idx, svg_path in enumerate(svg_files):
        if has_unsupported_features(svg_path):
            skipped += 1
            continue

        out_name = f"{idx:05d}_{svg_path.stem}.pt"
        out_path = precomputed_dir / out_name
        if out_path.exists() and not args.overwrite:
            saved += 1
            continue

        try:
            svg_tensor = svg_converter.svg_to_tensor(str(svg_path))
            image = svg_to_image(svg_path, args.image_size)
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                pixel_cls = outputs.last_hidden_state[:, 0].cpu().squeeze(0)

            payload = {
                "full_svg_matrix_content": svg_tensor,
                "final_pixel_cls_token": pixel_cls,
                "element_row_counts_for_stages": None,
                "svg_path": str(svg_path),
                "filename": svg_path.stem,
                "caption": caption_from_filename(svg_path.stem),
            }
            torch.save(payload, out_path)
            saved += 1
        except Exception as e:
            skipped += 1
            print(f"Skip {svg_path.name}: {e}")

    print(f"Done. saved={saved}, skipped={skipped}, out_dir={precomputed_dir}")


if __name__ == "__main__":
    main()
