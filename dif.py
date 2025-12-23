import argparse
import math
import os
import random
from dataclasses import dataclass
from glob import glob
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


class ImageFolderRecursive(Dataset):
    def __init__(self, root: str, transform):
        self.root = root
        self.transform = transform
        paths: List[str] = []
        for ext in IMG_EXTS:
            paths.extend(glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        self.paths = sorted(set(paths))
        if not self.paths:
            raise ValueError(f"No images found under: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./ldm_out")

    p.add_argument("--pretrained_vae", type=str, default="stabilityai/sd-vae-ft-mse")

    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every_epochs", type=int, default=1)
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_sample_images", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def sample_images(unet, vae, scheduler, device, out_path, num_images=8, num_steps=30):
    unet.eval()
    vae.eval()

    # Latent shape: (B, 4, H/8, W/8) for SD-like VAEs
    latent_h = vae.config.sample_size if hasattr(vae.config, "sample_size") and vae.config.sample_size else None
    # We'll infer from out_path resolution by using unet.sample_size (latent grid size)
    # but easiest is: assume downsample factor 8 relative to training resolution
    # We'll set shape from unet.config.sample_size if present.
    sample_size = unet.config.sample_size
    if isinstance(sample_size, (tuple, list)):
        lh, lw = sample_size
    else:
        lh = lw = int(sample_size)

    latents = torch.randn((num_images, unet.config.in_channels, lh, lw), device=device)
    scheduler.set_timesteps(num_steps, device=device)

    for t in scheduler.timesteps:
        model_out = unet(latents, t).sample
        latents = scheduler.step(model_out, t, latents).prev_sample

    # Decode to pixels (note: SD VAEs use a scaling_factor)
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    imgs = vae.decode(latents / scaling).sample  # [-1, 1]
    imgs = (imgs.clamp(-1, 1) + 1) / 2          # [0, 1]

    grid = make_grid(imgs, nrow=int(math.sqrt(num_images)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)
    unet.train()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    device = accelerator.device
    print(device)
    # Data: normalize to [-1, 1] (important for SD-like VAEs)
    tfm = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = ImageFolderRecursive(args.data_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print("data loaded")

    # Load a pretrained VAE (frozen)
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae)
    vae.requires_grad_(False)
    vae.to(device)
    print("VAE loaded")
    # Latent size is downsampled by factor ~8 (common for SD VAEs)
    latent_size = args.resolution // 8
    in_channels = 4  # SD-like latent channels
    unet = UNet2DModel(
        sample_size=latent_size,
        in_channels=in_channels,
        out_channels=in_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    print("UNET loaded")
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    unet, optimizer, dl = accelerator.prepare(unet, optimizer, dl)

    global_step = 0
    for epoch in range(args.epochs):
        unet.train()
        pbar = tqdm(dl, disable=not accelerator.is_local_main_process)
        pbar.set_description(f"epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            # batch: (B, 3, H, W) in [-1, 1]
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    posterior = vae.encode(batch.to(device)).latent_dist
                    latents = posterior.sample()

                    # Scale latents like Stable Diffusion VAEs do
                    scaling = getattr(vae.config, "scaling_factor", 1.0)
                    latents = latents * scaling

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if accelerator.is_local_main_process and (global_step % args.log_every == 0):
                pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        # Save + sample
        if accelerator.is_local_main_process:
            # Save UNet weights
            save_dir = os.path.join(args.output_dir, f"unet_epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            accelerator.unwrap_model(unet).save_pretrained(save_dir)

            if (epoch + 1) % args.sample_every_epochs == 0:
                sample_path = os.path.join(args.output_dir, "samples", f"epoch_{epoch+1:03d}.png")
                # Use unwrap_model for sampling on main process
                sample_images(
                    accelerator.unwrap_model(unet),
                    vae,
                    noise_scheduler,
                    device=device,
                    out_path=sample_path,
                    num_images=args.num_sample_images,
                    num_steps=args.num_sample_steps,
                )

    if accelerator.is_local_main_process:
        print(f"Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()