from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.app import fpn_mobilenet

# --- Your model with trainable sharpening layer ---
model = fpn_mobilenet(fpn, head1, head2, head3, head4, smooth, smooth2, final)
model.load_state_dict(torch.load(
    "your_pretrained_model.pth", map_location="cpu"))
model.to("cpu")  # or "cuda" if available

# Loss: pixel + perceptual
pixel_loss = nn.L1Loss()

# Optional: VGG perceptual loss
vgg = vgg16(pretrained=True).features[:16].eval()
for p in vgg.parameters():
    p.requires_grad = False


def perceptual_loss(pred, target):
    pred_vgg = vgg((pred + 1) / 2)  # to [0,1]
    target_vgg = vgg((target + 1) / 2)
    return nn.functional.l1_loss(pred_vgg, target_vgg)


# Optimizer — only fine-tune final + sharpening layer
params_to_train = list(model.final.parameters()) + \
    list(model.edge_enhancer.parameters())
optimizer = optim.Adam(params_to_train, lr=1e-4)

# Example DataLoader (replace with yours)
# dataset = YourDataset(...)
# loader = DataLoader(dataset, batch_size=4, shuffle=True)

epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for blurred, sharp in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        blurred, sharp = blurred.to("cpu"), sharp.to("cpu")

        optimizer.zero_grad()
        output = model(blurred, strength=1.0, edge_boost=0.3)

        loss_pix = pixel_loss(output, sharp)
        loss_perc = perceptual_loss(output, sharp)

        loss = loss_pix + 0.05 * loss_perc
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {epoch_loss / len(loader):.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), "deblur_finetuned.pth")
