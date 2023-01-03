import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from .TinyResNet18 import BasicBlock as oldbasic
from typing import List
from torch.distributions import Bernoulli


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


class OldBasic(oldbasic):
    def __init__(self, in_planes: int, planes: int, stride: int=1, args=None):
        super(OldBasic, self).__init__(in_planes, planes, stride)

    def forward(self, x:torch.Tensor, warmup=False, threshold=0.5):
        if warmup:
            return super().forward(x)
        else:
            if self.training:
                return super().forward(x), [], [], []
            else:
                return super().forward(x), []


class Sigmoid(nn.Sigmoid):
    def __init__(self, temperature):
        super(Sigmoid, self).__init__()
        self.temperature = temperature

    def forward(self, input):
        return torch.sigmoid(input / self.temperature)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1, args=None):
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.args = args
        self.gate1 = Agent(
            out_channels=planes,
            in_channels=in_planes,
            infer_mode=self.args.infer_mode,
            args=self.args
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.gate2 = Agent(
            out_channels=planes,
            in_channels=planes,
            infer_mode=self.args.infer_mode,
            args=self.args
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor, warmup=False, threshold=0.5) -> tuple:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        if not warmup:
            if self.training:
                actions1, policy_logits1, log_probs1 = self.gate1(x)
            else:
                actions1 = self.gate1(x, threshold=threshold)
            out = relu(self.bn1(self.conv1(x)) * actions1)
            if self.training:
                actions2, policy_logits2, log_probs2 = self.gate2(out)
            else:
                actions2 = self.gate2(out, threshold=threshold)
            out = self.bn2(self.conv2(out)) * actions2
            out += (self.shortcut(x)) * actions2
            out = relu(out)
            if self.training:
                return out, \
                       [actions1.squeeze(), actions2.squeeze()], \
                       [policy_logits1.squeeze(), policy_logits2.squeeze()], \
                       [log_probs1, log_probs2]
            else:
                return out, [actions1.squeeze(dim=-1).squeeze(dim=-1), actions2.squeeze(dim=-1).squeeze(dim=-1)]
        else:
            out = relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += (self.shortcut(x))
            out = relu(out)
            return out


class Agent(nn.Module):
    """
    The gate block for each convolutions.
    """

    def __init__(self, out_channels: int, in_channels: int=None,
                 infer_mode="sample", args=None):
        """
        :param out_channels:
        :param in_channels:
        :param infer_mode:
        """
        super(Agent, self).__init__()

        self.args = args

        self.channel_net = SelfAttention(
            out_channels=out_channels,
            in_channels=in_channels,
            hidden_dim_ratio=self.args.hidden_dim_ratio
        )

        self.action_layer = Sigmoid(self.args.temperature)
        self.infer_mode = infer_mode

    def forward(self, x: torch.Tensor, threshold=0.5):
        """

        :param x:
        :return:
        """
        policy_logits = self.channel_net(x)
        out = self.action_layer(policy_logits)

        if self.training or self.infer_mode == "sample":
            try:
                dist = Bernoulli(out)
                actions = dist.sample()
            except:
                print("Found Nan")
                exit()
        else:
            out[out < threshold] = 0.0
            out[out >= threshold] = 1.0
            actions = out

        if self.training:
            log_probs = dist.log_prob(actions)
            return actions, policy_logits, log_probs.squeeze().sum(dim=1, keepdim=True)
        else:
            return actions


class SelfAttention(nn.Module):
    """
    The gate block for each convolutions.
    """

    def __init__(self, out_channels: int, in_channels: int, hidden_dim_ratio: int, ):
        """
        :param out_channels:
        :param in_channels:
        :param infer_mode:
        """
        super(SelfAttention, self).__init__()
        self.pw_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prototype = nn.Sequential(
            nn.Linear(out_channels, out_channels // hidden_dim_ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // hidden_dim_ratio, out_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """

        :param x:
        :return:
        """
        out = self.bn(self.pw_conv(x))
        out = avg_pool2d(out, out.shape[2])
        N, C, _, _ = out.shape
        y = out.view(N, C)
        y = self.prototype(y).view(N, C, 1, 1)
        out = out * y
        return out


class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block, num_blocks: List[int],
                 num_classes: int, nf: int, args) -> None:
        """

        :param block:
        :param num_blocks:
        :param num_classes:
        :param nf:
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
        self.args = args

        firstblock = OldBasic

        strides = [1] + [2] * (len(num_blocks) - 1)
        self.layers = []
        self.agent_boundaries = [0]
        for i, stride in enumerate(strides):
            self.layers.append(self._make_layer(block if i else firstblock,
                                                nf * 2 ** i,
                                                num_blocks[i],
                                                stride=stride))
            self.agent_boundaries += [self.agent_boundaries[-1] + nf * 2 ** i * (j + 1) for j in range(4)]

        self.agent_boundaries = [agent_boundary - self.agent_boundaries[4]
                                 for agent_boundary in self.agent_boundaries[4:]]
        self.linear = nn.Linear(nf * (2 ** (len(num_blocks) - 1)) * block.expansion, num_classes)

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
                args=self.args
            ))
            if s_no == 0:
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, threshold=0.5) -> tuple:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """

        if not self.warmup:
            actions = []
            if self.training:
                policy_logits = []
                log_probs = []

        out = relu(self.bn1(self.conv1(x)))

        if not self.warmup:

            for layer in self.layers:
                for block in layer:
                    if self.training:
                        out, actions_block, policy_logits_block, log_probs_block = block(out)
                        actions += actions_block
                        policy_logits += policy_logits_block
                        log_probs += log_probs_block
                    else:
                        out, actions_block = block(out, threshold=threshold)
                        actions += actions_block
        else:
            for layer in self.layers:
                for block in layer:
                    out = block(out, warmup=True)

        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        if not self.warmup:
            if self.training:
                return out, torch.cat(actions, dim=1), torch.cat(policy_logits, dim=1), torch.cat(log_probs, dim=1)
            else:
                return out, torch.cat(actions, dim=1)
        else:
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

    def get_warmup(self) -> bool:
        """
        Returns warmup state of network training.
        :return: warmup state of network training
        """
        return self.warmup

    def set_warmup(self, warmup: bool):
        """
        :param warmup:new warmup state of network
        :return: None
        """
        self.warmup = warmup


def gatedresnet(nclasses: int, num_blocks: tuple = (2, 2), args=None) -> ResNet:
    """
    :param nclasses:
    :param num_blocks:
    :param args:
    :return:
    """
    block = BasicBlock
    return ResNet(block, list(num_blocks), nclasses, args.nf, args=args)
