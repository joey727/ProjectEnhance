"""
FP N + MobileNetV2 generator integration (PyTorch)
Save as: fpn_mobilenet_integration.py

Usage (example):
    from fpn_mobilenet_integration import load_model, deblur_image
    model = load_model("best_mobilenet.pth", device="cpu")
    deblur_image(model, "blur.jpg", "restored.jpg", device="cpu")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np


# ----------------- Utilities -----------------
def clean_state_dict(sd):
    """Strip common prefixes from keys (module., netG., etc.)."""
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("netG."):
            new_k = new_k[len("netG."):]
        new_sd[new_k] = v
    return new_sd


def load_checkpoint_state(ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    # ensure tensor objects (if someone saved numpy arrays)
    sd = {k: (torch.tensor(v) if isinstance(v, np.ndarray) else v)
          for k, v in sd.items()}
    return clean_state_dict(sd)


def pad_to_multiple(tensor, mult=32):
    """Pad H and W up to the next multiple of `mult`. Returns (tensor, (pad_h, pad_w))"""
    b, c, h, w = tensor.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)
    # F.pad uses (left, right, top, bottom)
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (pad_h, pad_w)


def preprocess_image_pil(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    return tensor


def deprocess_tensor_to_pil(tensor):
    """tensor: (1,C,H,W) in [-1,1]"""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr + 1.0) / 2.0
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------- FPN (MobileNetV2 backbone) -----------------
class FPNMobile(nn.Module):
    """
    Feature Pyramid Network using MobileNetV2 features.
    The chosen feature indices correspond to strides: 1/4, 1/8, 1/16, 1/32.
    """

    def __init__(self, norm_layer=nn.BatchNorm2d, num_filters=256, pretrained_backbone=True):
        super().__init__()
        mb = models.mobilenet_v2(pretrained=pretrained_backbone).features

        # Split MobileNetV2 features into blocks so we can get intermediate maps:
        # The splits below were chosen so outputs come roughly at scales 1/4,1/8,1/16,1/32
        # (works with torchvision's mobilenet_v2).
        # conv -> channels = 32 (stride=2)  -> ~1/2
        self.enc0 = nn.Sequential(mb[0])
        # includes the block that gives stride=2 -> ~1/4
        self.enc1 = nn.Sequential(*mb[1:3])
        self.enc2 = nn.Sequential(*mb[3:5])         # ~1/8
        self.enc3 = nn.Sequential(*mb[5:8])         # ~1/16
        self.enc4 = nn.Sequential(*mb[8:15])        # ~1/32

        # lateral 1x1 convs to unify channels into num_filters
        # channel counts are from MobileNetV2 typical layout (torchvision):
        # enc0 -> 32, enc1 -> 24, enc2 -> 32, enc3 -> 64, enc4 -> 160
        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(
            32, num_filters // 2, kernel_size=1, bias=False)

        # top-down convs to smooth after addition
        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        # freeze backbone by default
        for p in self.parameters():
            p.requires_grad = True  # we set True here; caller can call unfreeze() if desired
        for p in mb.parameters():
            p.requires_grad = False

        self.pad = nn.ReflectionPad2d(1)

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        # unfreeze backbone separately if needed:
        # for param in self.enc0.parameters(): param.requires_grad = True
        # etc.

    # def forward(self, x):
    #     # Encoder stages
    #     enc0 = self.enc0(x)     # ~1/2 resolution
    #     enc1 = self.enc1(enc0)  # ~1/4
    #     enc2 = self.enc2(enc1)  # ~1/8
    #     enc3 = self.enc3(enc2)  # ~1/16
    #     enc4 = self.enc4(enc3)  # ~1/32

    #     # Lateral connections
    #     lateral4 = self.pad(self.lateral4(enc4))
    #     lateral3 = self.pad(self.lateral3(enc3))
    #     lateral2 = self.lateral2(enc2)
    #     lateral1 = self.pad(self.lateral1(enc1))
    #     lateral0 = self.lateral0(enc0)

    #     # Top-down pathway
    #     map4 = lateral4

    #     map3 = self.td1(
    #         lateral3 +
    #         F.interpolate(map4, size=lateral3.shape[2:], mode="nearest")
    #     )

    #     # Pad both before addition if one is padded
    #     lat2_padded = F.pad(lateral2, (1, 2, 1, 2), mode="reflect")
    #     map3_up = F.interpolate(
    #         map3, size=lat2_padded.shape[2:], mode="nearest")
    #     map2 = self.td2(lat2_padded + map3_up)

    #     map1 = self.td3(
    #         lateral1 +
    #         F.interpolate(map2, size=lateral1.shape[2:], mode="nearest")
    #     )

    #     # Final output
    #     map0 = F.pad(lateral0, (0, 1, 0, 1), mode="reflect")

    #     return map0, map1, map2, map3, map4

    def forward(self, x):
        enc0 = self.enc0(x)          # ~1/2
        enc1 = self.enc1(enc0)      # ~1/4
        enc2 = self.enc2(enc1)      # ~1/8
        enc3 = self.enc3(enc2)      # ~1/16
        enc4 = self.enc4(enc3)      # ~1/32

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        # top-down (match sizes explicitly)
        map4 = lateral4

        map3 = self.td1(
            lateral3 +
            F.interpolate(map4, size=lateral3.shape[2:], mode="nearest")
        )

        map2 = self.td2(
            F.pad(lateral2, (1, 2, 1, 2), mode="reflect") +
            F.interpolate(map3, size=(
                F.pad(lateral2, (1, 2, 1, 2)).shape[2:]), mode="nearest")
        )

        map1 = self.td3(
            lateral1 +
            F.interpolate(map2, size=lateral1.shape[2:], mode="nearest")
        )

        return F.pad(lateral0, (0, 1, 0, 1), mode="reflect"), map1, map2, map3, map4


# ----------------- Generator using FPNMobile -----------------
class FPNMobileGenerator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, output_ch=3, num_filters=128, num_filters_fpn=256):
        super().__init__()
        self.fpn = FPNMobile(
            norm_layer=norm_layer, num_filters=num_filters_fpn, pretrained_backbone=True)

        # segmentation heads (reduce FPN features to num_filters)
        self.head1 = nn.Conv2d(num_filters_fpn, num_filters,
                               kernel_size=3, padding=1, bias=False)
        self.head2 = nn.Conv2d(num_filters_fpn, num_filters,
                               kernel_size=3, padding=1, bias=False)
        self.head3 = nn.Conv2d(num_filters_fpn, num_filters,
                               kernel_size=3, padding=1, bias=False)
        self.head4 = nn.Conv2d(num_filters_fpn, num_filters,
                               kernel_size=3, padding=1, bias=False)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(inplace=True),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(
            num_filters // 2, output_ch, kernel_size=3, padding=1)

    # def forward(self, x):
    #     map0, map1, map2, map3, map4 = self.fpn(x)

    #     # Process with heads
    #     map4 = self.head4(map4)
    #     map3 = self.head3(map3)
    #     map2 = self.head2(map2)
    #     map1 = self.head1(map1)

    #     # Step 1: Pick a reference size (map1 is usually fine)
    #     target_size = map1.size()[2:]  # (H, W)

    #     # Step 2: Force all maps to same size before concatenation
    #     map4 = F.interpolate(map4, size=target_size, mode="nearest")
    #     map3 = F.interpolate(map3, size=target_size, mode="nearest")
    #     map2 = F.interpolate(map2, size=target_size, mode="nearest")
    #     # map1 is already target_size

    #     smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))

    #     # Step 3: Upsample smoothed and match to map0 for addition
    #     smoothed = F.interpolate(smoothed, scale_factor=2, mode="nearest")
    #     map0 = F.interpolate(map0, size=smoothed.size()[2:], mode="nearest")

    #     smoothed = self.smooth2(smoothed + map0)

    #     # Step 4: Final upsample
    #     smoothed = F.interpolate(smoothed, scale_factor=2, mode="nearest")

    #     final = self.final(smoothed)

    #     # Match final to original input size
    #     final = F.interpolate(final, size=x.size()[2:], mode="nearest")

    #     # res = torch.tanh(final) + x
    #     # Without skip connection
    #     res = x + torch.tanh(final)

    #     # Or scale the residual
    #     # res = x + 0.5 * torch.tanh(final)

    #     # skip tanh
    #     # res = x + final

    def forward(self, x, strength=3.0):
        """
        Args:
            x: input tensor [-1, 1]
            strength: multiplier for how aggressively to apply deblurring changes
                    0.0 = no change, 1.0 = normal, >1.0 = more aggressive
        """
        map0, map1, map2, map3, map4 = self.fpn(x)

        # Heads
        map4 = self.head4(map4)
        map3 = self.head3(map3)
        map2 = self.head2(map2)
        map1 = self.head1(map1)

        # Reference size (map1)
        target_size = map1.size()[2:]
        map4 = F.interpolate(map4, size=target_size,
                             mode="bilinear", align_corners=False)
        map3 = F.interpolate(map3, size=target_size,
                             mode="bilinear", align_corners=False)
        map2 = F.interpolate(map2, size=target_size,
                             mode="bilinear", align_corners=False)

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))

        # Match sizes for addition with map0
        smoothed = F.interpolate(
            smoothed, scale_factor=2, mode="bilinear", align_corners=False)
        map0 = F.interpolate(map0, size=smoothed.size()[
                             2:], mode="bilinear", align_corners=False)

        smoothed = self.smooth2(smoothed + map0)

        # Final upsample
        smoothed = F.interpolate(
            smoothed, scale_factor=2, mode="bilinear", align_corners=False)

        final = self.final(smoothed)

        # Match to input size
        final = F.interpolate(final, size=x.size()[
                              2:], mode="bilinear", align_corners=False)

        # Apply tunable strength to the residual
        residual = torch.tanh(final) * strength
        res = x + residual

        return torch.clamp(res, min=-1, max=1)


# ----------------- Loader / Inference helpers -----------------
def load_model(weights_path=None, device="cpu"):
    """
    Instantiate the generator and (optionally) load a checkpoint.
    weights_path can be a .pth/.pt checkpoint (PyTorch).
    """
    device = torch.device(device)
    model = FPNMobileGenerator(norm_layer=nn.BatchNorm2d,
                               output_ch=3, num_filters=128, num_filters_fpn=256)
    model.to(device)

    if weights_path is not None:
        sd = load_checkpoint_state(weights_path, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("Loaded checkpoint:", weights_path)
        print("Missing keys (may be ok):", missing)
        print("Unexpected keys (may be ok):", unexpected)
    model.eval()
    return model


def deblur_image(model, input_path, output_path, device="cpu"):
    device = torch.device(device)
    model.to(device)
    inp = preprocess_image_pil(input_path).to(device)
    inp, pads = pad_to_multiple(inp, mult=32)
    with torch.no_grad():
        out = model(inp)
    if pads != (0, 0):
        pad_h, pad_w = pads
        h_cut = out.shape[2] - pad_h
        w_cut = out.shape[3] - pad_w
        out = out[:, :, :h_cut, :w_cut]
    out_img = deprocess_tensor_to_pil(out)
    out_img.save(output_path)
    print(f"Saved deblurred image to: {output_path}")


# ----------------- CLI test -----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python fpn_mobilenet_integration.py <input.jpg> <output.jpg> [weights.pth]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    weights = sys.argv[3] if len(sys.argv) == 4 else None
    model = load_model(weights, device="cpu")
    deblur_image(model, input_path, output_path, device="cpu")
