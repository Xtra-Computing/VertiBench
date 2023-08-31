import torch
import torch.nn as nn
from config import cfg
from torchvision.models import resnet18
from .utils import init_param, normalize, loss_fn, feature_split
from .interm import interm
from .late import late
from .vfl import vfl
from .dl import dl
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet18_VB(nn.Module):
    def __init__(self, data_shape, target_size, in_channels, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        output = {}
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        output['target'] = out
        if 'target' in input:
            if 'loss_mode' in input:
                output['loss'] = loss_fn(output['target'], input['target'], loss_mode=input['loss_mode'])
            else:
                output['loss'] = loss_fn(output['target'], input['target'])
        return output
    
    def feature(self, input):
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def resnet18_vb():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    if cfg['assist_mode'] == 'late':
        model = late(ResNet18_VB(data_shape, target_size))
    elif cfg['assist_mode'] in ['none', 'bag', 'stack']:
        if 'dl' in cfg and cfg['dl'] == '1':
            model = dl(ResNet18_VB(data_shape, target_size), target_size)
        else:
            if cfg['data_name'] == "CIFAR10_VB":
                model = ResNet18_VB(data_shape, target_size, 3 * cfg['num_users']) # 因为数据集是在 axis=1 拼接的，比如 (60000,1,28,28) 有4party时会拼成 (60000,4,28,28)
            elif cfg['data_name'] == "MNIST_VB":
                model = ResNet18_VB(data_shape, target_size, 1 * cfg['num_users']) # 再比如 (50000, 3, 32,32) 会拼成 (50000, 12, 32, 32)
            else:
                raise ValueError('Not valid data_name, please check your dataset')
    else:
        raise ValueError('Not valid assist mode')
    model.apply(init_param)
    return model
