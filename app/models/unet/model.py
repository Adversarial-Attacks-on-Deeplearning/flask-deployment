import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super().__init__()
        self.downs = nn.ModuleList()

        self.ups_transpose = nn.ModuleList()
        self.ups_conv = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down path of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups_transpose.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups_conv.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def down(self, x, skip_connections):
        """
        This method performs the downsampling (contracting) part of the U-Net.
        It appends the feature map to skip_connections and pools the output.
        """
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x

    def up(self, x, skip_connections):
        """
        This method performs the upsampling (expanding) part of the U-Net.
        It applies ConvTranspose2d followed by DoubleConv at each step.
        """
        skip_connections = skip_connections[::-
                                            1]
        for idx in range(len(self.ups_transpose)):
            x = self.ups_transpose[idx](x)
            skip_connection = skip_connections[idx]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)

            x = self.ups_conv[idx](concat_skip)

        return x

    def forward(self, x):
        skip_connections = []
        x = self.down(x, skip_connections)
        x = self.bottleneck(x)
        x = self.up(x, skip_connections)
        return self.final_conv(x)
