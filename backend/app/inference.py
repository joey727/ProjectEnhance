import torch
from PIL import Image
import torchvision.transforms as T

preprocess = T.Compose([
    T.ToTensor(),
])

postprocess = T.ToPILImage()


def center_crop_to_multiple(img: Image.Image, multiple: int = 32) -> Image.Image:
    w, h = img.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom))


def deblur_image(input_path, output_path, model):
    img = Image.open(input_path).convert("RGB")
    img = center_crop_to_multiple(img, 32)  # Ensure size multiple of 32

    transform = T.ToTensor()
    inp = transform(img).unsqueeze(0)  # shape: (1, C, H, W)

    with torch.no_grad():
        out = model(inp)

    out_img = T.ToPILImage()(out.squeeze(0))
    out_img.save(output_path)
