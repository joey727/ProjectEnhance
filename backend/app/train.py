"""
Progressive fine-tuning for FPN-MobileNet on GoPro (PyTorch)
------------------------------------------------------------
- Auto-detects GoPro folder structure:
    GoPro/
      train/ blurred/, sharp/
      test/  blurred/, sharp/
- Trains in progressive stages from --start_res to --final_res
- Uses L1 + Perceptual loss; optional GAN (PatchGAN + LSGAN)
- Pads inputs to multiple-of-32, unpads for loss/metrics
- Mixed precision + cosine LR per stage + checkpointing per stage

Run example:
  python train_fpn_mobilenet_progressive.py \
    --data_dir /path/to/GoPro \
    --weights /path/to/your_pretrained.pth \
    --start_res 128 --final_res 512 \
    --epochs_per_stage 10 --batch_size 4 \
    --adv_weight 0.01

It will write stage checkpoints into: runs/progressive_mobilenet/
"""

import torchvision.transforms as T
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import argparse
import os
from pathlib import Path
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image

# Your generator implementation must be available as a module
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


def unpad_to_original(x: torch.Tensor, pads: Tuple[int, int]):
    pad_h, pad_w = pads
    if pad_h == 0 and pad_w == 0:
        return x
    return x[:, :, : x.shape[2] - pad_h, : x.shape[3] - pad_w]


def psnr(pred: torch.Tensor, target: torch.Tensor):
    pred_01 = (pred.clamp(-1, 1) + 1.0) / 2.0
    tgt_01 = (target.clamp(-1, 1) + 1.0) / 2.0
    mse = F.mse_loss(pred_01, tgt_01)
    if mse.item() == 0:
        return 99.0
    return 10 * math.log10(1.0 / mse.item())


# ------------------------------ Data ------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


# class PairedByNameDataset(Dataset):
#     """Paired dataset expects two subfolders: blurred/ and sharp/ with matching filenames."""

#     def __init__(self, root_dir: Path, target_res: int, augment: bool = True):
#         self.blur_dir = root_dir / "blurred"
#         self.sharp_dir = root_dir / "sharp"
#         assert self.blur_dir.is_dir() and self.sharp_dir.is_dir(), (
#             f"Expected {self.blur_dir} and {self.sharp_dir} to exist"
#         )
#         blur_names = {p.name for p in list_images(self.blur_dir)}
#         sharp_names = {p.name for p in list_images(self.sharp_dir)}
#         self.names = sorted(list(blur_names & sharp_names))
#         if len(self.names) == 0:
#             raise RuntimeError(
#                 "No paired images found (matching filenames) in blurred/ and sharp/.")

#         self.target_res = int(target_res)
#         self.augment = augment

#         # Normalization to [-1, 1]
#         self.to_tensor = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Lambda(lambda t: t * 2.0 - 1.0),
#         ])

#     def __len__(self):
#         return len(self.names)

#     def resize_keep_aspect(self, img: Image.Image, short_side: int) -> Image.Image:
#         w, h = img.size
#         if min(w, h) == short_side:
#             return img
#         if w < h:
#             new_w = short_side
#             new_h = int(round(h * short_side / w))
#         else:
#             new_h = short_side
#             new_w = int(round(w * short_side / h))
#         return img.resize((new_w, new_h), Image.BICUBIC)

#     def __getitem__(self, idx):
#         name = self.names[idx]
#         blur_img = Image.open(self.blur_dir / name).convert("RGB")
#         sharp_img = Image.open(self.sharp_dir / name).convert("RGB")

#         # Progressive resizing: resize so short side >= target_res, then random crop target_res x target_res
#         blur_img = self.resize_keep_aspect(blur_img, self.target_res)
#         sharp_img = self.resize_keep_aspect(sharp_img, self.target_res)

#         # Ensure same spatial crop parameters
#         i, j, h, w = transforms.RandomCrop.get_params(
#             blur_img, (self.target_res, self.target_res))
#         blur_img = transforms.functional.crop(blur_img, i, j, h, w)
#         sharp_img = transforms.functional.crop(sharp_img, i, j, h, w)

#         if self.augment:
#             # Synchronized flips
#             if torch.rand(1) < 0.5:
#                 blur_img = transforms.functional.hflip(blur_img)
#                 sharp_img = transforms.functional.hflip(sharp_img)
#             if torch.rand(1) < 0.5:
#                 blur_img = transforms.functional.vflip(blur_img)
#                 sharp_img = transforms.functional.vflip(sharp_img)
#             # Small rotation, same angle
#             angle = (torch.rand(1).item() - 0.5) * 6  # -3..3 degrees
#             blur_img = transforms.functional.rotate(
#                 blur_img, angle, expand=False)
#             sharp_img = transforms.functional.rotate(
#                 sharp_img, angle, expand=False)

#         blur = self.to_tensor(blur_img)
#         sharp = self.to_tensor(sharp_img)
#         return blur, sharp, name


def load_and_transform(img_path, transform=None, target_res=None):
    img = Image.open(img_path).convert("RGB")
    if target_res is not None:
        img = img.resize(target_res, Image.BICUBIC)
    if transform:
        img = transform(img)
    return img


class PairedByNameDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_res=None, augment=False):
        self.root_dir = root_dir
        self.target_res = target_res
        self.augment = augment

        # Default transform ensures tensors are always returned
        self.transform = transform or T.Compose([
            T.ToTensor()
        ])

        self.blur_dir = os.path.join(root_dir, "blurred")
        self.sharp_dir = os.path.join(root_dir, "sharp")

        self.pairs = []
        sharp_files = sorted(os.listdir(self.sharp_dir))
        blur_files = sorted(os.listdir(self.blur_dir))
        for filename in sharp_files:
            if filename in blur_files:
                self.pairs.append((
                    os.path.join(self.blur_dir, filename),
                    os.path.join(self.sharp_dir, filename),
                    filename
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path, filename = self.pairs[idx]

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        # Resize
        if self.target_res:
            size = (self.target_res, self.target_res) if isinstance(
                self.target_res, int) else self.target_res
            blur_img = blur_img.resize(size, Image.BICUBIC)
            sharp_img = sharp_img.resize(size, Image.BICUBIC)

        # Augmentations
        if self.augment:
            if random.random() > 0.5:
                blur_img = ImageOps.mirror(blur_img)
                sharp_img = ImageOps.mirror(sharp_img)
            if random.random() > 0.5:
                blur_img = ImageOps.flip(blur_img)
                sharp_img = ImageOps.flip(sharp_img)

        # Convert to tensors here
        blur_img = self.transform(blur_img)
        sharp_img = self.transform(sharp_img)

        return blur_img, sharp_img, filename


# ------------------------------ Losses ------------------------------


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss using relu1_2, relu2_2, relu3_3 features."""

    def __init__(self, weight: float = 1.0, device: str = "cpu"):
        super().__init__()
        # Try to load torchvision weights; fallback to random if not available
        vgg = vgg16(weights=None)
        try:
            sd = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/vgg16-397923af.pth', progress=True
            )
            vgg.load_state_dict(sd)
        except Exception:
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
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
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


# ------------------------------ Training per stage ------------------------------

def make_loaders(data_dir: Path, res: int, batch_size: int, workers: int):
    train_ds = PairedByNameDataset(
        data_dir / 'train', target_res=res, augment=True)
    val_ds = PairedByNameDataset(
        data_dir / 'test', target_res=res, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader


def train_stage(netG, netD, optG, optD, losses, loaders, device, args, stage_idx, res):
    l1_loss, perc_loss, gan_loss = losses
    train_loader, val_loader = loaders

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == 'cuda')
    schedG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optG, T_max=args.epochs_per_stage)

    use_gan = args.adv_weight > 0.0 and netD is not None and optD is not None

    best_psnr = -1e9
    for epoch in range(1, args.epochs_per_stage + 1):
        netG.train()
        if use_gan:
            netD.train()

        running = {"l1": 0.0, "perc": 0.0, "gan_g": 0.0, "gan_d": 0.0}

        for blur, sharp, _ in train_loader:
            blur = blur.to(device, non_blocking=True)
            sharp = sharp.to(device, non_blocking=True)

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
                    logits_fake = netD((pred + 1) / 2.0)
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
            else:
                d_total = torch.tensor(0.0, device=device)

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

        avg_l1 = running['l1'] / max(1, len(train_loader))
        avg_perc = running['perc'] / max(1, len(train_loader))
        print(f"[Stage {stage_idx} | {res}px] Epoch {epoch:03d}/{args.epochs_per_stage} | L1 {avg_l1:.4f} | "
              f"Perc {avg_perc:.4f} | PSNR {val_psnr:.2f} dB | LR {schedG.get_last_lr()[0]:.2e}")

        # Save per-epoch checkpoint
        stage_dir = Path(args.out_dir) / f"stage_{res}px"
        stage_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            'epoch': epoch,
            'netG': netG.state_dict(),
            'optG': optG.state_dict(),
            'res': res,
            'args': vars(args),
        }
        torch.save(ckpt, stage_dir / f"ckpt_epoch_{epoch:03d}.pth")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(netG.state_dict(), stage_dir / "best_netG.pth")
            print(
                f"✓ [Stage {stage_idx} | {res}px] New best → {stage_dir/'best_netG.pth'} ({val_psnr:.2f} dB)")

    return best_psnr


# ------------------------------ CLI / Main ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True,
                   help='Root GoPro dir with train/ and test/')
    p.add_argument('--weights', type=str, default='',
                   help='Optional pretrained generator weights (.pth)')
    p.add_argument('--out_dir', type=str, default='runs/progressive_mobilenet')

    p.add_argument('--start_res', type=int, default=128)
    p.add_argument('--final_res', type=int, default=512)
    p.add_argument('--epochs_per_stage', type=int, default=10)

    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--amp', action='store_true',
                   help='Use mixed precision training')
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU even if CUDA is available')
    p.add_argument('--seed', type=int, default=1337)

    # Loss weights
    p.add_argument('--l1_weight', type=float, default=1.0)
    p.add_argument('--perc_weight', type=float, default=1.0)
    p.add_argument('--adv_weight', type=float, default=0.0,
                   help='Set >0 to enable GAN training (e.g., 0.01)')

    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.cpu else 'cpu')

    data_dir = Path(args.data_dir)
    # Sanity check structure
    for split in ['train', 'test']:
        if not (data_dir / split / 'blurred').is_dir() or not (data_dir / split / 'sharp').is_dir():
            raise SystemExit(
                f"Expected {data_dir}/{split}/blurred and {data_dir}/{split}/sharp")

    # Model
    netG = FPNMobileGenerator().to(device)
    if args.weights and os.path.isfile(args.weights):
        sd = torch.load(args.weights, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
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
    netD = NLayerDiscriminator().to(device) if use_gan else None
    optD = torch.optim.AdamW(
        netD.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999)) if use_gan else None

    # Losses
    l1_loss = nn.L1Loss()
    perc_loss = PerceptualLoss(weight=args.perc_weight, device=device)
    gan_loss = GANLoss() if use_gan else None

    # Optimizer for G (shared across stages)
    optG = torch.optim.AdamW(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Build progressive resolution ladder
    res_list = []
    r = max(32, args.start_res)
    final_r = max(r, args.final_res)
    while r < final_r:
        res_list.append(r)
        r *= 2
    if res_list and res_list[-1] != final_r:
        res_list.append(final_r)
    if not res_list:
        res_list = [final_r]

    print(f"Progressive stages: {res_list}")

    os.makedirs(args.out_dir, exist_ok=True)
    for si, res in enumerate(res_list, start=1):
        # Rebuild loaders for this stage & possibly adjust batch size
        stage_bs = args.batch_size
        if device.type == 'cuda':
            # heuristic: reduce batch size for larger resolutions
            if res >= 512:
                stage_bs = max(1, args.batch_size // 2)
            if res >= 768:
                stage_bs = max(1, args.batch_size // 4)
        print(f"\n=== Stage {si} @ {res}px | batch_size={stage_bs} ===")
        train_loader, val_loader = make_loaders(
            data_dir, res, stage_bs, args.workers)

        best = train_stage(
            netG=netG,
            netD=netD,
            optG=optG,
            optD=optD,
            losses=(l1_loss, perc_loss, gan_loss),
            loaders=(train_loader, val_loader),
            device=device,
            args=args,
            stage_idx=si,
            res=res,
        )
        print(f"Stage {si} best PSNR: {best:.2f} dB")

    # Save final generator
    torch.save(netG.state_dict(), Path(args.out_dir) / "final_netG.pth")
    print(
        f"\n✓ Training complete. Final weights → {Path(args.out_dir) / 'final_netG.pth'}")


if __name__ == "__main__":
    main()
