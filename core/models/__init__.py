import torch
import torch.nn as nn
import torch.nn.functional as F

from .cifar10.standard_resnet import standard_resnet as standard_resnet_cifar10
from .cifar10.pdeadd_resnet import pdeadd_resnet as pdeadd_resnet_cifar10

from .cifar100.standard_resnet import standard_resnet as standard_resnet_cifar100
from .cifar100.pdeadd_resnet import pdeadd_resnet as pdeadd_resnet_cifar100

from .tinyin200.standard_resnet import standard_resnet as standard_resnet_tinyin200
from .tinyin200.pdeadd_resnet import pdeadd_resnet as pdeadd_resnet_tinyin200


def create_model(data, backbone, protocol):

    net = backbone.split('-')[0]
    if 'fixdiff' in protocol: 
        protocol, sigma = protocol.split('-')
    if 'ladiff' in protocol: 
        protocol, _ = protocol.split('-')

    model_name = "{}_{}_{}".format(protocol, net, data)
    print("using name: {}..".format(model_name))
    func = eval(model_name)

    if 'fixdiff' in protocol: 
        return func(backbone, float(sigma))
    else:
        return func(backbone)
