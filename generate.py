"""
SVG 생성 스크립트 - 완전 구현
"""

import torch
from pathlib import Path
import argparse
from transformers import CLIPTokenizer, CLIPTextModel

from models import VPVAE, VSDiT
from utils import DiffusionUtils
from utils.svg_parser import SVGToTensor
from config import VAEConfig, DiTConfig


def generate_svg(args):
    """SVG 생성"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # VAE 로드
    print("Loading VAE...")
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    vae_config_dict = vae_checkpoint.get('config', {})
    
    vae_config = VAEConfig()
    for k, v in vae_config_dict.items():
        if hasattr(vae_config, k):
            setattr(vae_config, k, v)
    
    vae = VPVAE(vae_config).to(device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    print(f"Loaded VAE from: {args.vae_checkpoint}")
    
    # DiT 로드
    print("Loading DiT...")
    dit_checkpoint = torch.load(args.dit_checkpoint, map_location=device)
    dit_config_dict = dit_checkpoint.get('config', {})
    
    dit_config = DiTConfig()
    for k, v in dit_config_dict.items():
        if hasattr(dit_config, k):
            setattr(dit_config, k, v)
    
    dit = VSDiT(dit_config).to(device)
    dit.load_state_dict(dit_checkpoint['model_state_dict'])
    dit.eval()
    print(f"Loaded DiT from: {args.dit_checkpoint}")
    
    # CLIP 텍스트 인코더
    print("Loading CLIP...")
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    clip_text_encoder = CLIPTextModel.from_pretrained(args.clip_model).to(device)
    clip_text_encoder.eval()
    
    # Diffusion 유틸리티
    diffusion = DiffusionUtils(
        noise_steps=dit_config.noise_steps,
        beta_start=dit_config.beta_start,
        beta_end=dit_config.beta_end
    )
    
    # SVG 변환기
    svg_converter = SVGToTensor(max_seq_len=vae_config.max_seq_len)
    
    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 텍스트 임베딩
    print(f"\nGenerating SVGs for prompt: '{args.prompt}'")
    
    with torch.no_grad():
        # Conditional 텍스트 임베딩
        text_inputs = clip_tokenizer(
            [args.prompt] * args.num_samples,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        
        text_outputs = clip_text_encoder(**text_inputs)
        cond_context = text_outputs.last_hidden_state  # [num_samples, seq_len, D]
        
        # Unconditional 임베딩 (CFG용)
        uncond_inputs = clip_tokenizer(
            [""] * args.num_samples,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        
        uncond_outputs = clip_text_encoder(**uncond_inputs)
        uncond_context = uncond_outputs.last_hidden_state
        
        # 초기 노이즈
        latent_shape = (args.num_samples, vae_config.max_seq_len, dit_config.latent_dim)
        latents = torch.randn(latent_shape, device=device)
        
        # 마스크 (전체 valid)
        latent_mask = torch.zeros((args.num_samples, vae_config.max_seq_len), 
                                 dtype=torch.bool, device=device)
        
        print(f"Starting DDIM sampling ({args.ddim_steps} steps, CFG scale: {args.cfg_scale})...")
        
        # DDIM 샘플링
        latents = diffusion.ddim_sample(
            model=dit,
            latent_shape=latent_shape,
            conditional_context=cond_context,
            unconditional_context=uncond_context,
            cfg_scale=args.cfg_scale,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
            latent_mask=latent_mask,
            device=device
        )
        
        print("Decoding latents to SVG tensors...")
        
        # VAE 디코딩
        outputs = vae.decode(latents, latent_mask)
        
        # 텐서를 SVG 파일로 변환
        print(f"Saving SVG files to {output_dir}...")
        
        for i in range(args.num_samples):
            # 각 예측에서 가장 높은 확률의 값 선택
            element_ids = outputs['element_logits'][i].argmax(dim=-1)  # [L]
            command_ids = outputs['command_logits'][i].argmax(dim=-1)  # [L]
            param_values = outputs['param_logits'][i].round().clamp(0, 255).long()  # [L, 12]
            
            # SVG 텐서 구성
            svg_tensor = torch.cat([
                element_ids.unsqueeze(-1),
                command_ids.unsqueeze(-1),
                param_values
            ], dim=-1)  # [L, 14]
            
            # 파일명
            prompt_slug = args.prompt.replace(' ', '_')[:30]
            output_file = output_dir / f"{prompt_slug}_sample{i+1}.svg"
            
            # SVG 저장
            try:
                svg_converter.tensor_to_svg(
                    tensor=svg_tensor,
                    output_file=str(output_file),
                    width=512,
                    height=512
                )
                print(f"  Saved: {output_file}")
            except Exception as e:
                print(f"  Error saving {output_file}: {e}")
    
    print(f"\nGeneration complete! {args.num_samples} SVG files saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SVG from text")
    
    # 모델 체크포인트
    parser.add_argument('--vae_checkpoint', type=str, required=True, help='VAE checkpoint path')
    parser.add_argument('--dit_checkpoint', type=str, required=True, help='DiT checkpoint path')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model')
    
    # 생성 파라미터
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--cfg_scale', type=float, default=7.0, help='Classifier-free guidance scale')
    parser.add_argument('--ddim_steps', type=int, default=100, help='Number of DDIM steps')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta (0=deterministic)')
    
    # 출력
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    generate_svg(args)
