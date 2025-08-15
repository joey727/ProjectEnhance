import torch
from backend.app.fpn_mobilenet import FPNMobileGenerator


def load_model(weights_path, device="cpu"):
    model = FPNMobileGenerator()  # match constructor params
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
