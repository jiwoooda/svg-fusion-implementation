import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import VPVAE
from config import VAEConfig, PAD_IDX


class PrecomputedSVGDataset(Dataset):
    def __init__(self, precomputed_dir: str, max_files: int = 0):
        self.precomputed_dir = Path(precomputed_dir)
        self.files = sorted(self.precomputed_dir.glob("*.pt"))
        if max_files and max_files > 0:
            self.files = self.files[:max_files]
        if not self.files:
            raise ValueError(f"No .pt files found in {precomputed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        payload = torch.load(self.files[idx], map_location="cpu")
        return {
            "svg_tensor": payload["full_svg_matrix_content"].long(),
            "pixel_cls": payload["final_pixel_cls_token"].float(),
            "filename": payload.get("filename", self.files[idx].stem),
        }


def collate_precomputed(batch):
    max_len = max(item["svg_tensor"].shape[0] for item in batch)
    num_params = batch[0]["svg_tensor"].shape[1]
    pixel_dim = batch[0]["pixel_cls"].shape[0]

    batch_svg = []
    batch_pixel = []
    batch_mask = []
    batch_filenames = []

    for item in batch:
        svg = item["svg_tensor"]
        seq_len = svg.shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            pad_row = torch.zeros((pad_len, num_params), dtype=torch.long)
            pad_row[:, 0] = PAD_IDX
            svg_padded = torch.cat([svg, pad_row], dim=0)
            mask = torch.cat(
                [torch.zeros(seq_len, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)]
            )
        else:
            svg_padded = svg
            mask = torch.zeros(seq_len, dtype=torch.bool)

        pixel_cls = item["pixel_cls"].view(1, pixel_dim)
        pixel_seq = pixel_cls.repeat(max_len, 1)
        if pad_len > 0:
            pixel_seq[seq_len:] = 0.0

        batch_svg.append(svg_padded)
        batch_pixel.append(pixel_seq)
        batch_mask.append(mask)
        batch_filenames.append(item["filename"])

    return {
        "svg_tensor": torch.stack(batch_svg),
        "pixel_embedding": torch.stack(batch_pixel),
        "svg_mask": torch.stack(batch_mask),
        "filename": batch_filenames,
    }


def save_checkpoint(path: Path, model, optimizer, step, loss, config):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "loss": loss,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke-test VAE training")
    parser.add_argument("--precomputed_dir", type=str, required=True, help="Cache directory")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ckpt_out", type=str, default="checkpoints", help="Checkpoint output dir")
    parser.add_argument("--max_files", type=int, default=0, help="Limit number of cache files")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Grad clip")
    parser.add_argument("--recon_mse_loss_weight", type=float, default=1.0, help="Recon loss weight")
    parser.add_argument("--kl_anneal_portion", type=float, default=0.5, help="KL anneal portion")
    parser.add_argument("--kl_weight_max", type=float, default=0.5, help="Max KL weight")
    parser.add_argument("--log_every", type=int, default=50, help="Log interval")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PrecomputedSVGDataset(args.precomputed_dir, max_files=args.max_files)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_precomputed,
        drop_last=False,
    )

    sample = dataset[0]
    pixel_dim = sample["pixel_cls"].shape[0]

    config = VAEConfig()
    config.pixel_embed_dim = pixel_dim

    model = VPVAE(
        num_element_types=config.num_element_types,
        num_command_types=config.num_command_types,
        element_embed_dim=config.element_embed_dim,
        command_embed_dim=config.command_embed_dim,
        num_continuous_params=config.num_continuous_params,
        pixel_feature_dim=config.pixel_embed_dim,
        encoder_d_model=config.encoder_d_model,
        decoder_d_model=config.decoder_d_model,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        num_heads=config.num_heads,
        latent_dim=config.latent_dim,
        max_seq_len=config.max_seq_len,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_out = Path(args.ckpt_out)
    ckpt_out.mkdir(parents=True, exist_ok=True)

    model.train()
    data_iter = iter(dataloader)
    best_loss = math.inf
    total_steps = max(args.steps, 1)
    anneal_steps = int(total_steps * args.kl_anneal_portion)

    for step in range(1, total_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        svg_tensor = batch["svg_tensor"].to(device)
        pixel_emb = batch["pixel_embedding"].to(device)
        svg_mask = batch["svg_mask"].to(device)

        element_ids = svg_tensor[:, :, 0]
        command_ids = svg_tensor[:, :, 1]
        continuous_params = svg_tensor[:, :, 2:]

        outputs = model(
            svg_element_ids=element_ids,
            svg_command_ids=command_ids,
            svg_continuous_params=continuous_params,
            pixel_features=pixel_emb,
            svg_mask=svg_mask,
        )

        predicted = outputs["predicted_features"]
        target = model.normalize_target(element_ids, command_ids, continuous_params)

        L_out = predicted.size(1)
        L_tgt = target.size(1)
        effective_len = min(L_out, L_tgt)

        preds_eff = predicted[:, :effective_len, :]
        targets_eff = target[:, :effective_len, :]
        mask_eff = svg_mask[:, :effective_len]

        feature_valid_mask = (~mask_eff).unsqueeze(-1).expand_as(targets_eff).float()
        num_valid = feature_valid_mask.sum() + 1e-9

        mse_unreduced = nn.functional.mse_loss(
            preds_eff * feature_valid_mask, targets_eff * feature_valid_mask, reduction="none"
        )
        mse_recon = mse_unreduced.sum() / num_valid

        kl_loss = outputs["kl_loss"]
        if step < anneal_steps:
            kl_weight = args.kl_weight_max * step / max(1, anneal_steps)
        else:
            kl_weight = args.kl_weight_max

        loss = args.recon_mse_loss_weight * mse_recon + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step}/{total_steps} "
                f"loss={loss.item():.4f} recon={mse_recon.item():.4f} "
                f"kl={kl_loss.item():.4f} kl_w={kl_weight:.3f} lr={lr:.6f}"
            )

        if step % args.save_every == 0 or step == total_steps:
            ckpt_path = ckpt_out / f"model_step{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, loss.item(), config)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_path = ckpt_out / "model_best.pt"
            save_checkpoint(best_path, model, optimizer, step, loss.item(), config)

    print(f"Training done. best_loss={best_loss:.4f} ckpt_dir={ckpt_out}")


if __name__ == "__main__":
    main()
