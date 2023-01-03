import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int=1):
    """
    :param in_planes:
    :param out_planes:
    :param stride:
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class Sigmoid(nn.Sigmoid):
    def __init__(self, temperature):
        super(Sigmoid, self).__init__()
        self.temperature = temperature

    def forward(self, input):
        return torch.sigmoid(input / self.temperature)


class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """

        :param block:
        :param num_blocks:
        :param num_classes:
        :param nf:
        :param channel_imp:
        :param infer_mode:
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)

        strides = [1, 2]
        self.layers = []
        self.agent_boundaries = [0]
        for i, stride in enumerate(strides):
            self.layers.append(self._make_layer(block,
                                                nf * 2 ** i,
                                                num_blocks[i],
                                                stride=stride))
        self.linear = nn.Linear(nf * 2 * block.expansion, num_classes)

        self._features = nn.Sequential(*[self.conv1, self.bn1, nn.ReLU()] + self.layers)
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s_no, stride in enumerate(strides):
            layers.append(block(
                self.in_planes, planes, stride,
            ))
            if s_no == 0:
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """

        out = relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)

        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)
        return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def tinyresnet(nclasses: int, nf: int = 32) -> ResNet:
    """
    :param nclasses:
    :param nf:
    :param args:
    :return:
    """
    return ResNet(BasicBlock, [2, 2], nclasses, nf)
