import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from fpn_inception import FPNInception


def clean_state_dict(sd):
    """Strip common prefixes ('module.', 'netG.', etc.) from keys."""
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        # common DataParallel prefix
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        # sometimes checkpoints store model under 'netG.' prefix
        if new_k.startswith("netG."):
            new_k = new_k[len("netG."):]
        new_sd[new_k] = v
    return new_sd


def load_fpn_inception_weights(model, ckpt_path, device='cpu'):
    """
    Load weights from the official/community checkpoint. Handles a few
    common checkpoint formats encountered in forks (dict with 'model' key,
    state_dict directly, etc.).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    # ckpt might be a dict containing 'model' or 'state_dict' keys
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            sd = ckpt['model']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
    else:
        sd = ckpt  # sometimes the raw state_dict was saved

    sd = clean_state_dict(sd)
    # Attempt load (use strict=False to allow small mismatches)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def preprocess_image_pil(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    # model output uses tanh and residual -> training used inputs in [-1,1]
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    return tensor


def pad_to_multiple(tensor, mult=4):
    # tensor shape: (B,C,H,W)
    b, c, h, w = tensor.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)
    pad = (0, pad_w, 0, pad_h)  # (left,right,top,bottom)
    tensor = F.pad(tensor, pad, mode='reflect')
    return tensor, (pad_h, pad_w)


def deprocess_tensor_to_image(tensor):
    """tensor: (1,C,H,W) in model output space [-1,1]"""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # H,W,C
    arr = (arr + 1.0) / 2.0  # -> [0,1]
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# Example usage:
if __name__ == "__main__":
    device = torch.device('cpu')  # use cpu; change to 'cuda' if available
    # instantiate model (match the repo call)
    model = FPNInception(norm_layer=torch.nn.BatchNorm2d,
                         output_ch=3, num_filters=128, num_filters_fpn=256)
    model.to(device)

    # load weights (path to your downloaded file)
    ckpt_path = "backend/app/best_fpn.h5"
    missing, unexpected = load_fpn_inception_weights(
        model, ckpt_path, device=device)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    model.eval()
    img_path = "test_blurry.jpg"
    inp = preprocess_image_pil(img_path)
    inp, pads = pad_to_multiple(inp, mult=4)
    with torch.no_grad():
        out = model(inp.to(device))
    # if padded, crop back
    if pads != (0, 0):
        pad_h, pad_w = pads
        h_cut = out.shape[2] - pad_h
        w_cut = out.shape[3] - pad_w
        out = out[:, :, :h_cut, :w_cut]
    out_img = deprocess_tensor_to_image(out)
    out_img.save("deblurred_output.jpg")
    print("saved deblurred_output.jpg")
