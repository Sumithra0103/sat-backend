import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet18, ResNet18_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SiameseResNet18ChangeDetector(nn.Module):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # Shared pretrained ResNet-18 encoder
        # --------------------------------------------------

        if HAS_TORCHVISION:
            try:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                backbone = resnet18(weights=None)
        else:
            raise ImportError("torchvision is required for SiameseResNet18ChangeDetector")

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # --------------------------------------------------
        # ImageNet normalization for pretrained backbone
        # --------------------------------------------------

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1),
        )

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        self.dec4 = ConvBlock(
            512,
            256
        )

        self.dec3 = ConvBlock(
            256 + 256,
            128
        )

        self.dec2 = ConvBlock(
            128 + 128,
            64
        )

        self.dec1 = ConvBlock(
            64 + 64,
            32
        )

        self.dec0 = ConvBlock(
            32 + 64,
            32
        )

        self.head = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    # ------------------------------------------------------
    # Encoder
    # ------------------------------------------------------

    def encode(self, x):

        x = (x - self.mean) / self.std

        s = self.stem(x)

        x = self.maxpool(s)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return s, f1, f2, f3, f4

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    def forward(self, t1, t2):

        s1, f11, f21, f31, f41 = self.encode(t1)
        s2, f12, f22, f32, f42 = self.encode(t2)

        # Feature differences
        d0 = torch.abs(s1 - s2)
        d1 = torch.abs(f11 - f12)
        d2 = torch.abs(f21 - f22)
        d3 = torch.abs(f31 - f32)
        d4 = torch.abs(f41 - f42)

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        x = self.dec4(d4)

        x = F.interpolate(
            x,
            size=d3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, d3],
            dim=1
        )

        x = self.dec3(x)

        x = F.interpolate(
            x,
            size=d2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, d2],
            dim=1
        )

        x = self.dec2(x)

        x = F.interpolate(
            x,
            size=d1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, d1],
            dim=1
        )

        x = self.dec1(x)

        x = F.interpolate(
            x,
            size=d0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, d0],
            dim=1
        )

        x = self.dec0(x)

        # Back to original 256x256
        x = F.interpolate(
            x,
            size=t1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return self.head(x)


if __name__ == "__main__":

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = SiameseResNet18ChangeDetector().to(device)

    x1 = torch.randn(
        2, 3, 256, 256,
        device=device
    )

    x2 = torch.randn(
        2, 3, 256, 256,
        device=device
    )

    with torch.no_grad():
        output = model(x1, x2)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Device:", device)
    print("Parameters:", f"{parameters:,}")
    print("Input:", tuple(x1.shape))
    print("Output:", tuple(output.shape))
