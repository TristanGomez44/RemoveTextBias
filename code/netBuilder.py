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
import math
from PIL import Image

class GeomLayer(nn.Module):

    def __init__(self,inChan,inSize,chan):

        super(GeomLayer, self).__init__()
        self.geoParams = nn.Parameter(torch.rand((chan,2,3)).float())
        self.linComb = nn.Parameter(torch.rand(chan,inChan))

        self.chan = chan

        self.boxParams = nn.Parameter(torch.rand((chan,2,3)).float())

    def forward(self,x):


        linComb = self.linComb.unsqueeze(0).unsqueeze(3).unsqueeze(4)
        linComb = linComb.expand(x.size(0),self.chan,x.size(1),x.size(2),x.size(3))

        x = x.unsqueeze(1).expand(x.size(0),self.chan,x.size(1),x.size(2),x.size(3))

        x = (x*linComb).sum(dim=2,keepdim=True)

        box = self.applyAffTrans(torch.ones_like(x),self.boxParams)
        x = self.applyAffTrans(x,self.geoParams)

        finalPred = x*box
        return finalPred,x,box

    def applyAffTrans(self,x,geoParams):

        geoParams = geoParams.unsqueeze(0).expand(x.size(0),self.chan,2,3)
        geoParams = geoParams.contiguous().view(x.size(0)*self.chan,2,3)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        grid = F.affine_grid(geoParams, x.size())
        x = F.grid_sample(x, grid)
        x = x.view(x.size(0)//self.chan,self.chan,x.size(2),x.size(3))

        return x
        #template = self.template.unsqueeze(0).expand(x.size(0),self.template.size(0),self.template.size(1),self.template.size(2))

        #x = F.relu(x*template)

def netMaker(args):
    '''Build a network
    Args:
        args (Namespace): the namespace containing all the arguments required for training and building the network
    Returns:
        the built network
    '''

    if args.dataset == "IMAGENET":
        net = resnet.resnet18(pretrained=False,geom=args.geom)

        stateDict = torch.load("../nets/resnet18_imageNet.pth")

        for key in stateDict.keys():
            if not key.endswith("conv2.weight") or (not args.geom):
                net.state_dict()[key].data += stateDict[key].data - net.state_dict()[key].data

    elif args.dataset == "MNIST" or args.dataset == "CIFAR10":

        if args.dataset == "MNIST":
            inSize = 28
            inChan = 1
        else:
            inSize = 32
            inChan = 3

        numClasses = 10

        net = resnet.ResNet(resnet.BasicBlock, [1, 1, 1, 1],geom=args.geom,inChan=inChan, width_per_group=args.dechan,maxpool=False,\
                                strides=[1,2,2,2],firstConvKer=args.deker,inPlanes=args.dechan,num_classes=numClasses)

    else:
        raise ValueError("Unknown dataset : ".format(args.dataset))

    return net

if __name__ == "__main__":

    def preprocc(x):
        #print(x.size())
        x = ((x-x.min())/(x.max()-x.min())).permute(1,2,0).squeeze()

        return (x.detach().numpy()*255).astype('uint8')

    np.random.seed(0)
    torch.manual_seed(0)
    inChan = 3
    theta = math.pi/4
    target = torch.zeros(10,inChan,200,200)
    target[:,:,50:150,50:150] = 1

    Image.fromarray(preprocc(target[0])).save("../vis/testNetBuilder_targ.jpg")

    geomParam = torch.tensor([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0]]).unsqueeze(0).expand(target.size(0),2,3)

    grid = F.affine_grid(geomParam, target.size())
    x = F.grid_sample(target, grid)

    Image.fromarray(preprocc(x[0])).save("../vis/testNetBuilder_in.jpg")

    geom = GeomLayer(inChan,(200,200),inChan)

    opti = torch.optim.SGD(geom.parameters(), lr=0.05, momentum=0.9)

    for i in range(1000):
        opti.zero_grad()

        finalPred,pred,mask = geom(x)

        loss = torch.pow(finalPred-target,2).mean()

        loss.backward()

        opti.step()

        if i % 100 == 0:
            print(i)
            Image.fromarray(preprocc(finalPred[0])).save("../vis/testNetBuilder_finalPred_{}.jpg".format(i))
            Image.fromarray(preprocc(pred[0])).save("../vis/testNetBuilder_pred_{}.jpg".format(i))
            Image.fromarray(preprocc(mask[0])).save("../vis/testNetBuilder_mask_{}.jpg".format(i))
