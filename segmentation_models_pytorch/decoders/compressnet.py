import torch.nn as nn


class CompressionNet(nn.Module):
    def __init__(self, in_ch=1):
        super(CompressionNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ELU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ELU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=in_ch, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.relu3 = nn.ELU(inplace=True)

        # self.conv4 = nn.Conv2d(in_channels=8, out_channels=out_ch, kernel_size=5, stride=2, padding=2)
        # self.bn4 = nn.BatchNorm2d(3)
        # self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        # First convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Third convolution block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Fourth convolution block
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu4(x)

        return x
