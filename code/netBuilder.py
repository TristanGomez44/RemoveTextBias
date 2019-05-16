import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import str2bool
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import numpy as np

import resnet

def netMaker(args):
    '''Build a network
    Args:
        args (Namespace): the namespace containing all the arguments required for training and building the network
    Returns:
        the built network
    '''

    net = resnet.resnet18(pretrained=False,geom=args.geom)

    stateDict = torch.load("../nets/resnet18_imageNet.pth")

    for key in stateDict.keys():

        if not key.endswith("conv2.weight") or (not args.geom):
            net.state_dict()[key].data += stateDict[key].data - net.state_dict()[key].data

    return net
