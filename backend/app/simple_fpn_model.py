import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPN(nn.Module):
    """Simplified FPN (Feature Pyramid Network) without external dependencies."""

    def __init__(self, num_filters=256):
        super(SimpleFPN, self).__init__()

        # Simple encoder (replaces InceptionResNetV2)
        self.enc0 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Lateral connections
        self.lateral4 = nn.Conv2d(512, num_filters, 1)
        self.lateral3 = nn.Conv2d(256, num_filters, 1)
        self.lateral2 = nn.Conv2d(128, num_filters, 1)
        self.lateral1 = nn.Conv2d(64, num_filters, 1)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, 1)

        # Top-down pathway
        self.td1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.td2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.td3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Bottom-up pathway
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Lateral connections
        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + F.interpolate(map4,
                        scale_factor=2, mode='nearest'))
        map2 = self.td2(lateral2 + F.interpolate(map3,
                        scale_factor=2, mode='nearest'))
        map1 = self.td3(lateral1 + F.interpolate(map2,
                        scale_factor=2, mode='nearest'))

        return lateral0, map1, map2, map3, map4


class SimpleFPNInception(nn.Module):
    """Simplified FPNInception model without external dependencies."""

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):
        super(SimpleFPNInception, self).__init__()

        # Feature Pyramid Network
        self.fpn = SimpleFPN(num_filters=num_filters_fpn)

        # Segmentation heads
        self.head1 = nn.Sequential(
            nn.Conv2d(num_filters_fpn, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(num_filters_fpn, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(num_filters_fpn, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(num_filters_fpn, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        # Smoothing layers
        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, 3, padding=1),
            nn.BatchNorm2d(num_filters // 2),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final = nn.Conv2d(num_filters // 2, output_ch, 3, padding=1)

    def forward(self, x):
        # Get FPN features
        map0, map1, map2, map3, map4 = self.fpn(x)

        # Apply heads and upsample to match input size
        target_size = x.shape[2:]  # Get target size from input

        map4 = F.interpolate(self.head4(map4), size=target_size,
                             mode='bilinear', align_corners=False)
        map3 = F.interpolate(self.head3(map3), size=target_size,
                             mode='bilinear', align_corners=False)
        map2 = F.interpolate(self.head2(map2), size=target_size,
                             mode='bilinear', align_corners=False)
        map1 = F.interpolate(self.head1(map1), size=target_size,
                             mode='bilinear', align_corners=False)

        # Combine features
        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = F.interpolate(
            smoothed, size=target_size, mode='bilinear', align_corners=False)

        # Add map0 (which should be the same size as input)
        map0 = F.interpolate(map0, size=target_size,
                             mode='bilinear', align_corners=False)
        smoothed = self.smooth2(smoothed + map0)

        # Final output
        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)
