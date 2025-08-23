
import argparse
import os
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image

# Import your generator implementation
# Make sure this module exists and exposes FPNMobileGenerator
from fpn_mobilenet import FPNMobileGenerator  # noqa: E402


# ------------------------------ Utils ------------------------------
def seed_everything(seed: int = 1337):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    """Pad BCHW tensor (reflect) so H,W are divisible by `multiple`. Return (x_pad, (pad_h, pad_w))."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def unpad_to_original(x: torch.Tensor, pads: tuple[int, int]):
    pad_h, pad_w = pads
    if pad_h == 0 and pad_w == 0:
        return x
    return x[:, :, : x.shape[2] - pad_h, : x.shape[3] - pad_w]


# ------------------------------ Data ------------------------------
class PairedDeblurDataset(Dataset):
    """Paired dataset where file names match between blurred/ and sharp/ folders."""

    def __init__(self, root_dir: str, crop_size: int = 256, augment: bool = True):
        self.blur_dir = Path(root_dir) / "blurred"
        self.sharp_dir = Path(root_dir) / "sharp"
        # assert self.blur_dir.is_dir() and self.sharp_dir.is_dir(), (
        #     f"Expected {self.blur_dir} and {self.sharp_dir} to exist"
        # )
        # index by common filenames present in both
        blur_names = {p.name for p in self.blur_dir.iterdir() if p.suffix.lower() in {
            ".png", ".jpg", ".jpeg"}}
        sharp_names = {p.name for p in self.sharp_dir.iterdir() if p.suffix.lower() in {
            ".png", ".jpg", ".jpeg"}}
        self.names = sorted(list(blur_names & sharp_names))
        assert len(
            self.names) > 0, "No paired images found. Ensure filenames match in blurred/ and sharp/."

        t_list = []
        if augment:
            t_list += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(3),  # small rotation
            ]
        if crop_size > 0:
            t_list += [transforms.RandomCrop(crop_size)]
        self.aug = transforms.Compose(t_list) if t_list else None

        # to tensor in [-1,1]
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # [0,1]
            transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        blur_img = Image.open(self.blur_dir / name).convert("RGB")
        sharp_img = Image.open(self.sharp_dir / name).convert("RGB")
        if self.aug:
            # apply same augmentation by stacking and unstacking
            w = Image.fromarray(
                torch.cat([
                    transforms.ToTensor()(blur_img),
                    transforms.ToTensor()(sharp_img)
                ], dim=0).permute(1, 2, 0).numpy().astype('float32')
            )
            # The above is cumbersome; simpler: apply identical transforms by seeding
            # Instead, we do deterministic transform via functional API when needed.
            # For now, just apply random crop + flips independently but with same params:
            # We'll implement deterministic crop by sampling params from blur and applying to sharp.

        # Deterministic spatial augmentations
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            blur_img = transforms.functional.hflip(blur_img)
            sharp_img = transforms.functional.hflip(sharp_img)
        # Random vertical flip
        if torch.rand(1) < 0.5:
            blur_img = transforms.functional.vflip(blur_img)
            sharp_img = transforms.functional.vflip(sharp_img)
        # Small rotation (same angle)
        angle = (torch.rand(1).item() - 0.5) * 6  # -3..3 degrees
        blur_img = transforms.functional.rotate(blur_img, angle, expand=False)
        sharp_img = transforms.functional.rotate(
            sharp_img, angle, expand=False)

        # Random crop to min common size
        bw, bh = blur_img.size
        sw, sh = sharp_img.size
        tw, th = min(bw, sw), min(bh, sh)
        blur_img = transforms.functional.center_crop(blur_img, min(th, tw))
        sharp_img = transforms.functional.center_crop(sharp_img, min(th, tw))

        # Optional random crop size
        # If images are big, do a random crop of 256
        crop_size = min(256, min(blur_img.size[0], blur_img.size[1]))
        i, j, h, w = transforms.RandomCrop.get_params(
            blur_img, (crop_size, crop_size))
        blur_img = transforms.functional.crop(blur_img, i, j, h, w)
        sharp_img = transforms.functional.crop(sharp_img, i, j, h, w)

        blur = self.to_tensor(blur_img)
        sharp = self.to_tensor(sharp_img)
        return blur, sharp, name


# ------------------------------ Losses ------------------------------
class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss using relu1_2, relu2_2, relu3_3 features."""

    def __init__(self, weight: float = 1.0, device: str = "cpu"):
        super().__init__()
        vgg = vgg16(weights=None)
        # Load ImageNet weights if available locally via torchvision (optional but recommended)
        try:
            vgg = vgg16(weights=torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/vgg16-397923af.pth',
                progress=True
            ))
        except Exception:
            # fallback to randomly initialized features (still works but weaker)
            pass
        features = vgg.features.eval()
        for p in features.parameters():
            p.requires_grad = False
        self.blocks = nn.ModuleList([
            features[:4].to(device),    # relu1_2
            features[4:9].to(device),   # relu2_2
            features[9:16].to(device),  # relu3_3
        ])
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # inputs in [-1,1] -> [0,1] -> normalize
        def norm(x):
            x = (x + 1.0) / 2.0
            return (x - self.mean) / self.std
        x = norm(pred)
        y = norm(target)
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)
        return self.weight * loss


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (70x70 by default)."""

    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_ch, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                      stride=1, padding=padw)  # patch map
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    """LSGAN loss for stability."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred_logits, target_is_real: bool):
        target = torch.ones_like(
            pred_logits) if target_is_real else torch.zeros_like(pred_logits)
        return self.loss(pred_logits, target)


# ------------------------------ Training ------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor):
    # inputs in [-1,1]
    pred_01 = (pred.clamp(-1, 1) + 1.0) / 2.0
    tgt_01 = (target.clamp(-1, 1) + 1.0) / 2.0
    mse = F.mse_loss(pred_01, tgt_01)
    if mse.item() == 0:
        return 99.0
    return 10 * math.log10(1.0 / mse.item())


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")

    # Data
    train_ds = PairedDeblurDataset(
        args.train_dir, crop_size=args.crop, augment=True)
    val_ds = PairedDeblurDataset(
        args.val_dir, crop_size=args.crop, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    netG = FPNMobileGenerator().to(device)
    if args.weights and os.path.isfile(args.weights):
        sd = torch.load(args.weights, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        # strip common prefixes
        cleaned = {}
        for k, v in sd.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('netG.'):
                k = k[5:]
            cleaned[k] = v
        missing, unexpected = netG.load_state_dict(cleaned, strict=False)
        print(
            f"Loaded weights: {args.weights}\nMissing: {len(missing)} Unexpected: {len(unexpected)}")

    # Discriminator (optional)
    use_gan = args.adv_weight > 0.0
    if use_gan:
        netD = NLayerDiscriminator().to(device)
        gan_loss = GANLoss().to(device)
        optD = torch.optim.AdamW(
            netD.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))

    # Losses
    l1_loss = nn.L1Loss()
    perc_loss = PerceptualLoss(weight=args.perc_weight, device=device)

    # Optim + sched
    optG = torch.optim.AdamW(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    schedG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optG, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == 'cuda')

    best_psnr = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        netG.train()
        if use_gan:
            netD.train()

        running = {"l1": 0.0, "perc": 0.0, "gan_g": 0.0, "gan_d": 0.0}

        for blur, sharp, _ in train_loader:
            blur = blur.to(device, non_blocking=True)
            sharp = sharp.to(device, non_blocking=True)

            # pad to multiple of 32 (match generator expectations)
            blur_pad, pads = pad_to_multiple(blur, 32)
            sharp_pad, _ = pad_to_multiple(sharp, 32)

            # ----------------- G step -----------------
            optG.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == 'cuda'):
                pred_pad = netG(blur_pad)
                pred = unpad_to_original(pred_pad, pads)

                l1 = l1_loss(pred, sharp)
                perc = perc_loss(pred, sharp)
                g_total = args.l1_weight * l1 + args.perc_weight * perc

                if use_gan:
                    logits_fake = netD((pred + 1) / 2.0)  # D expects [0,1]
                    g_gan = gan_loss(logits_fake, True)
                    g_total = g_total + args.adv_weight * g_gan
                else:
                    g_gan = torch.tensor(0.0, device=device)

            scaler.scale(g_total).backward()
            scaler.step(optG)
            scaler.update()

            # ----------------- D step -----------------
            if use_gan:
                optD.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == 'cuda'):
                    with torch.no_grad():
                        pred = pred.detach()
                    logits_real = netD((sharp + 1) / 2.0)
                    logits_fake = netD((pred + 1) / 2.0)
                    d_real = gan_loss(logits_real, True)
                    d_fake = gan_loss(logits_fake, False)
                    d_total = 0.5 * (d_real + d_fake)
                scaler.scale(d_total).backward()
                scaler.step(optD)

            # metrics
            running["l1"] += l1.item()
            running["perc"] += perc.item()
            running["gan_g"] += g_gan.item() if use_gan else 0.0
            running["gan_d"] += d_total.item() if use_gan else 0.0

        schedG.step()

        # ----------------- Validation -----------------
        netG.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for blur, sharp, _ in val_loader:
                blur = blur.to(device)
                sharp = sharp.to(device)
                blur_pad, pads = pad_to_multiple(blur, 32)
                pred_pad = netG(blur_pad)
                pred = unpad_to_original(pred_pad, pads)
                val_psnr += psnr(pred, sharp)
        val_psnr /= max(1, len(val_loader))

        print(f"Epoch {epoch:03d}/{args.epochs} | L1 {running['l1']/len(train_loader):.4f} | "
              f"Perc {running['perc']/len(train_loader):.4f} | "
              f"PSNR {val_psnr:.2f} dB | LR {schedG.get_last_lr()[0]:.2e}")

        # Save checkpoint
        is_best = val_psnr > best_psnr
        best_psnr = max(best_psnr, val_psnr)
        ckpt = {
            'epoch': epoch,
            'netG': netG.state_dict(),
            'optG': optG.state_dict(),
            'best_psnr': best_psnr,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(
            args.out_dir, f"ckpt_epoch_{epoch:03d}.pth"))
        if is_best:
            torch.save(netG.state_dict(), os.path.join(
                args.out_dir, "best_netG.pth"))
            print("✓ Saved new best generator weights → best_netG.pth")


# ------------------------------ CLI ------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', type=str, required=True,
                   help='Path with blurred/ and sharp/ for training')
    p.add_argument('--val_dir', type=str, required=True,
                   help='Path with blurred/ and sharp/ for validation')
    p.add_argument('--weights', type=str, default='',
                   help='Pretrained generator weights (.pth)')
    p.add_argument('--out_dir', type=str, default='runs/finetune_mobilenet')

    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--amp', action='store_true',
                   help='Use mixed precision training')
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU even if CUDA is available')
    p.add_argument('--seed', type=int, default=1337)

    p.add_argument('--crop', type=int, default=256)

    # Loss weights
    p.add_argument('--l1_weight', type=float, default=1.0)
    p.add_argument('--perc_weight', type=float, default=1.0)
    p.add_argument('--adv_weight', type=float, default=0.0,
                   help='Set >0 to enable GAN training (e.g., 0.01)')

    args = p.parse_args()
    train(args)
