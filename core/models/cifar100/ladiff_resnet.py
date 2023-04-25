"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion(nn.Module):
    def __init__(self, planes):
        super(Diffusion, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(planes, 2*planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(2*planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*planes, planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))

    def forward(self, input):
        out =  self.main(input)
        return out
    

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        self.diff = Diffusion(out_channels)

    def forward(self, x):
        x, use_diffusion, _ = x 
        out = nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
        if use_diffusion:
            sigma = self.diff(out)
            out = out + sigma * torch.randn_like(out)
            return (out, use_diffusion, sigma)
        else:
            return (out, use_diffusion, 0)



class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, use_diffusion = False):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = F.log_softmax(output,dim=1)

        return output

    def net(self, x, use_diffusion=True):
        
        self.mus = []
        self.sigmas = [] 
        self.scales = []

        out = self.conv1(x)
        out = self.conv2_x((out, use_diffusion, 0))
        self.mus.append(out[0])
        self.sigmas.append(out[2])
        if use_diffusion:
            self.scales.append(out[2].max().detach().data.item())
        else:
            self.scales.append(0)

        out = self.conv3_x(out)
        self.mus.append(out[0])
        self.sigmas.append(out[2])
        if use_diffusion:
            self.scales.append(out[2].max().detach().data.item())
        else:
            self.scales.append(0)

        out = self.conv4_x(out)
        self.mus.append(out[0])
        self.sigmas.append(out[2])
        if use_diffusion:
            self.scales.append(out[2].max().detach().data.item())
        else:
            self.scales.append(0)

        out = self.conv5_x(out)
        self.mus.append(out[0])
        self.sigmas.append(out[2])
        if use_diffusion:
            self.scales.append(out[2].max().detach().data.item())
        else:
            self.scales.append(0)

        out = out[0]

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    
    def forward(self, x, use_diffusion=True):
        if self.training:
            #print('training........')
            out = self.net(x, use_diffusion=use_diffusion)
            #out = F.log_softmax(out, dim=1)
        else:
            #print('evaling........')
            if use_diffusion:
                proba = 0 
                for k in range(10):  
                    out = self.net(x, use_diffusion=True)
                    #p = F.softmax(out, dim=1)
                    proba = proba + out
                out = proba/10 # next nll
            else:
                out = self.net(x, use_diffusion=False)
                #out = F.log_softmax(out, dim=1)
        return out



def ladiff_resnet(name, num_classes=100, pretrained=False, device='cpu'):
    print("cifar100..")
    """
    Returns suitable Resnet model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        pretrained (bool): whether to use a pretrained model.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if name == 'resnet-18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif name == 'resnet-34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    # elif name == 'resnet-50':
    #     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    # elif name == 'resnet-101':
    #     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    
    raise ValueError('Only resnet-18, resnet-34, resnet-50 and resnet-101 are supported!')
    return
