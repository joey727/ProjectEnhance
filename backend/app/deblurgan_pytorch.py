import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from backend.app.model_generator import FPNInception


class DeblurGANv2(nn.Module):
    def __init__(self):
        super(DeblurGANv2, self).__init__()

        # Use the FPNInception model from model_generator.py
        self.fpn_model = FPNInception(
            nn.BatchNorm2d, output_ch=3, num_filters=128, num_filters_fpn=256)

    def forward(self, x):
        return self.fpn_model(x)


class DeblurGANPredictorPyTorch:
    def __init__(self, model_path):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeblurGANv2().to(self.device)

        # Load the model weights
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint:
                    # Handle the case where weights are under 'model' key
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"PyTorch DeblurGAN model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
                print("Using untrained model (will produce poor results)")
        else:
            print(f"Model file not found: {model_path}")
            print("Using untrained model (will produce poor results)")

        self.model.eval()

    def predict(self, pil_image):
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Resize to 256x256
        pil_image = pil_image.resize((256, 256))

        # Convert to tensor and normalize to [-1, 1]
        img = np.array(pil_image).astype(np.float32)
        img = (img / 127.5) - 1.0
        img = torch.from_numpy(img).permute(
            2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Run inference
            output = self.model(img)

            # Convert back to numpy
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Convert from [-1, 1] to [0, 255]
            output = ((output + 1) * 127.5).astype(np.uint8)
            output = np.clip(output, 0, 255)

            return Image.fromarray(output)
