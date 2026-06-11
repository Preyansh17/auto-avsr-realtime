import torch
from torch import nn


def conv3x3_2d(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_1d(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_2d(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_1d(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm1d(outplanes),
    )


def activation(name: str, channels: int = None):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "prelu":
        return nn.PReLU(num_parameters=channels or 1)
    if name == "swish":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="swish"):
        super().__init__()
        self.conv1 = conv3x3_2d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = activation(relu_type, planes)
        self.conv2 = conv3x3_2d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = activation(relu_type, planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu2(out + residual)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="swish"):
        super().__init__()
        self.conv1 = conv3x3_1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu1 = activation(relu_type, planes)
        self.conv2 = conv3x3_1d(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu2 = activation(relu_type, planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu2(out + residual)


class ResNet2D(nn.Module):
    def __init__(self, block, layers, relu_type="swish"):
        super().__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_2d(self.inplanes, planes * block.expansion, stride)
        layers = [block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class ResNet1D(nn.Module):
    def __init__(self, block, layers, relu_type="swish", a_upsample_ratio=1):
        super().__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.a_upsample_ratio = a_upsample_ratio
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=80, stride=4, padding=38, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = activation(relu_type, self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=20 // a_upsample_ratio, stride=20 // a_upsample_ratio)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_1d(self.inplanes, planes * block.expansion, stride)
        layers = [block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x)


def three_d_to_two_d_tensor(x):
    batch, channels, frames, height, width = x.shape
    x = x.transpose(1, 2)
    return x.reshape(batch * frames, channels, height, width)


class Conv3dResNet(nn.Module):
    def __init__(self, relu_type="swish"):
        super().__init__()
        self.frontend_nout = 64
        self.trunk = ResNet2D(BasicBlock2D, [2, 2, 2, 2], relu_type=relu_type)
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            activation(relu_type, self.frontend_nout),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, xs_pad):
        xs_pad = xs_pad.transpose(2, 1)
        batch, _, _, _, _ = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)
        frames = xs_pad.shape[2]
        xs_pad = three_d_to_two_d_tensor(xs_pad)
        xs_pad = self.trunk(xs_pad)
        return xs_pad.view(batch, frames, xs_pad.size(1))


class Conv1dResNet(nn.Module):
    def __init__(self, relu_type="swish", a_upsample_ratio=1):
        super().__init__()
        self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type, a_upsample_ratio=a_upsample_ratio)

    def forward(self, xs_pad):
        batch, timesteps, channels = xs_pad.size()
        xs_pad = xs_pad[:, : timesteps // 640 * 640, :]
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = self.trunk(xs_pad)
        return xs_pad.transpose(1, 2)


def video_resnet():
    return Conv3dResNet()


def audio_resnet():
    return Conv1dResNet()
