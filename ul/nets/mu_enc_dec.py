import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, downsample=None, stride=1, momentum=0.1
    ):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = nn.functional.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks1 = nn.ModuleList(
            [
                ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum)
                for _ in range(1)
            ]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(
            out_channels // 2,
            out_channels,
            momentum=momentum,
            stride=2,
            downsample=self.conv2,
        )
        self.resblocks2 = nn.ModuleList(
            [
                ResidualBlock(out_channels, out_channels, momentum=momentum)
                for _ in range(1)
            ]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResidualBlock(out_channels, out_channels, momentum=momentum)
                for _ in range(1)
            ]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


# Encode the observations into hidden states
class Encoder(nn.Module):
    def __init__(
        self,
        inp_ch,
        inp_size,
        num_blocks=1,
        num_channels=64,
        downsample=True,
        momentum=0.1,
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        observation_shape = (inp_ch, inp_size, inp_size)
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        self.conv = conv3x3(
            observation_shape[0],
            num_channels,
        )
        self.resblocks = nn.ModuleList(
            [
                ResidualBlock(num_channels, num_channels, momentum=momentum)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


def main():
    C, D = 3, 84
    encoder = Encoder(inp_ch=C, inp_size=D)
    x = torch.randn((1, C, D, D))
    print(encoder(x).shape)
    # summary
    summary(
        encoder,
        (1, C, D, D),
        col_names=["output_size", "num_params", "kernel_size", "mult_adds"],
    )
    


if __name__ == "__main__":
    main()