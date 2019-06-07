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

from skimage.transform import resize
import matplotlib.pyplot as plt

class MaxPool2d_G(nn.Module):

    def __init__(self,kerSize):
        super(MaxPool2d_G, self).__init__()
        self.kerSize = kerSize

    def forward(self,x):

        convKerSize = x.size(-2)//self.kerSize[0],x.size(-1)//self.kerSize[1]

        #Compute the mean activation in every rectangle of size convKerSize with a convolution
        weight = (torch.ones((convKerSize)).unsqueeze(0).unsqueeze(0).expand(x.size(1),1,convKerSize[0],convKerSize[1])/(convKerSize[0]+convKerSize[1])).to(x.device)
        x_conv = torch.nn.functional.conv1d(x, weight,padding=(convKerSize[0]//2,convKerSize[1]//2),groups=x.size(1))[:,:,:x.size(-2),:x.size(-1)]

        #Get the position of the maxima for each feature map of each batch
        origImgSize = x.size(-2),x.size(-1)
        x_conv = x_conv.contiguous().view(x.size(0),x.size(1),x.size(2)*x.size(3))
        argm = torch.argmax(x_conv, dim=-1)
        argm = argm//origImgSize[0],argm%origImgSize[1]

        #Padd the input with zeros on the border
        paddedX = torch.zeros((x.size(0),x.size(1),x.size(2)+x.size(2)//(self.kerSize[0]),x.size(3)+x.size(3)//(self.kerSize[1]))).to(x.device)
        ctr = paddedX.size(-2)//2,paddedX.size(-1)//2
        offs = convKerSize[0],convKerSize[1]
        paddedX[:,:,ctr[0]-offs[0]:ctr[0]+offs[0]+x.size(-2)%2,ctr[1]-offs[1]:ctr[1]+offs[1]+x.size(-1)%2] = x

        #Compute the binary mask indicating the position of the selected pixels
        rowInds = torch.arange(paddedX.size(-2)).unsqueeze(1).expand(paddedX.size(-2),paddedX.size(-1)).to(x.device)
        colInds = torch.arange(paddedX.size(-1)).unsqueeze(0).expand(paddedX.size(-2),paddedX.size(-1)).to(x.device)
        rowInds = rowInds.unsqueeze(0).unsqueeze(0).expand(argm[0].size(0),argm[0].size(1),paddedX.size(-2),paddedX.size(-1))
        colInds = colInds.unsqueeze(0).unsqueeze(0).expand(argm[1].size(0),argm[1].size(1),paddedX.size(-2),paddedX.size(-1))

        argm = argm[0].unsqueeze(-1).unsqueeze(-1),argm[1].unsqueeze(-1).unsqueeze(-1)
        argm = argm[0]+x.size(2)//(2*self.kerSize[0]),argm[1]+x.size(3)//(2*self.kerSize[1])

        binaryArr = (argm[0]-convKerSize[0]//2 <= rowInds)*(rowInds <= argm[0]+convKerSize[0]//2-(convKerSize[0]%2==0))*(argm[1]-convKerSize[1]//2 <= colInds)*(colInds <= argm[1]+convKerSize[1]//2-(convKerSize[1]%2==0))

        #Selecting the pixels
        x = paddedX[binaryArr].view(x.size(0),x.size(1),convKerSize[0],convKerSize[1])

        return x

class GNN(nn.Module):

    def __init__(self,inChan,inSize,chan,nbLay,nbOut,resCon,batchNorm,maxPoolPos,maxPoolKer):

        super(GNN, self).__init__()

        self.inLay = GeomLayer(inChan,inSize,chan,batchNorm)
        self.resCon = resCon

        layList = []
        for i in range(nbLay+(not maxPoolPos is None)):

            if i == maxPoolPos:
                layList.append(MaxPool2d_G((maxPoolKer,maxPoolKer)))
            else:
                layList.append(GeomLayer(chan,inSize,chan,batchNorm))

        layers = nn.ModuleList(layList)

        self.layers = nn.Sequential(*layers)
        self.dense = nn.Linear(chan,nbOut)
        print(self)
    def forward(self,x):
        x = self.inLay(x)

        if self.resCon:
            for i in range(len(self.layers)):
                x = self.layers[i](x)+x
            featMaps = x
        else:
            featMaps = self.layers(x)

        x = featMaps.mean(dim=-1).mean(dim=-1)
        x = self.dense(x)

        return x,featMaps

class GeomLayer(nn.Module):

    def __init__(self,inChan,inSize,chan,batchNorm):

        super(GeomLayer, self).__init__()

        initVal = torch.eye(3)[:2].unsqueeze(0).expand(chan,2,3).float()
        noise = Variable(initVal.data.new(initVal.size()).normal_(0, 0.1))
        #initVal[:,0,2] = 0.75
        #initVal[:,1,2] = 0.75

        self.geoParams = nn.Parameter(initVal+noise)
        self.linComb = nn.Parameter(torch.rand(chan,inChan))

        self.chan = chan
        self.batchNorm = nn.BatchNorm2d(inChan) if batchNorm else None

        initVal = torch.eye(3)[:2].unsqueeze(0).expand(chan,2,3).float()
        #initVal[:,0,2] = 0.75
        #initVal[:,1,2] = 0.75
        noise = Variable(initVal.data.new(initVal.size()).normal_(0, 0.1))
        self.boxParams = nn.Parameter(initVal+noise)

    def forward(self,x):

        if not self.batchNorm is None:
            x = self.batchNorm(x)

        linComb = self.linComb.unsqueeze(0).unsqueeze(3).unsqueeze(4).to(x.device)
        linComb = linComb.expand(x.size(0),self.chan,x.size(1),x.size(2),x.size(3))

        x = x.unsqueeze(1).expand(x.size(0),self.chan,x.size(1),x.size(2),x.size(3))

        x = (x*linComb).sum(dim=2,keepdim=True)

        box = self.applyAffTrans(torch.ones_like(x),self.boxParams)
        x = self.applyAffTrans(x,self.geoParams)

        finalPred = x*box

        return finalPred

    def applyAffTrans(self,x,geoParams):

        geoParams = geoParams.unsqueeze(0).expand(x.size(0),self.chan,2,3).to(x.device)
        geoParams = geoParams.contiguous().view(x.size(0)*self.chan,2,3)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        grid = F.affine_grid(geoParams, x.size())

        #grid = torch.remainder(grid+1,2)-1

        xSamp = F.grid_sample(x, grid)
        xSamp = xSamp.view(xSamp.size(0)//self.chan,self.chan,xSamp.size(2),xSamp.size(3))

        return xSamp
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

    elif args.dataset == "MNIST" or args.dataset == "FAKENIST" or args.dataset == "CIFAR10":

        if args.dataset == "MNIST" or args.dataset == "FAKENIST":
            inSize = 28
            inChan = 1
        else:
            inSize = 32
            inChan = 3

        numClasses = 10

        if args.model == "cnn":
            net = resnet.ResNet(resnet.BasicBlock, [1, 1, 1, 1],geom=args.geom,inChan=inChan, width_per_group=args.dechan,maxpool=False,\
                                strides=[1,2,2,2],firstConvKer=args.deker,inPlanes=args.dechan,num_classes=numClasses)
        elif args.model == "gnn":
            net = GNN(inChan,inSize,args.chan_gnn,args.nb_lay_gnn,numClasses,args.res_con_gnn,args.batch_norm_gnn,args.max_pool_pos,args.max_pool_ker)

        else:
            raise ValueError("Unknown model type : {}".format(args.model))

    else:
        raise ValueError("Unknown dataset : {}".format(args.dataset))

    return net

if __name__ == "__main__":

    def preprocc(x):
        #print(x.size())
        if len(x.size()) == 3:
            x = ((x-x.min())/(x.max()-x.min())).permute(1,2,0).squeeze()
            npImg = (x.detach().numpy()*255)

            npImg = resize(npImg,(300,300),mode="constant", order=0,anti_aliasing=True).astype('uint8')

        else:
            x = ((x-x.min())/(x.max()-x.min()))
            npImg = (x.detach().numpy()*255)

            npImg = resize(npImg,(300,300),mode="constant", order=0,anti_aliasing=True).astype('uint8')

        return npImg

    def plotHeat(img,path):
        print(img.size())
        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.savefig(path)

    '''
    np.random.seed(0)
    torch.manual_seed(0)
    inChan = 1
    theta = math.pi/4
    tr = 0.75
    img1 = torch.zeros(10,inChan,200,200)
    img1[:,:,50:150,50:150] = 1
    #img1[:,:,5:10,5:10] = 1
    #img1[:,:,10:15,10:15] = 1

    plotHeat(img1[0,0],"../vis/testNet_gridImg1.png")

    geomParam = torch.eye(3)[:2]
    #initVal = torch.eye(3)[:2].unsqueeze(0).expand(chan,2,3).float()
    noise = Variable(geomParam.data.new(geomParam.size()).normal_(0, 0.3))

    #geomParam += noise

    geomParam[0,2] = tr
    geomParam[1,2] = tr

    geomParam = geomParam.unsqueeze(0).expand(img1.size(0),2,3)

    grid = F.affine_grid(geomParam, img1.size())

    plotHeat(grid[0,:,:,0],"../vis/testNet_gridx_heat.png")
    plotHeat(grid[0,:,:,1],"../vis/testNet_gridy_heat.png")

    img2 = F.grid_sample(img1, grid)

    plotHeat(img2[0,0],"../vis/testNet_gridImg2.png")

    grid = torch.remainder(grid+1,2)-1

    plotHeat(grid[0,:,:,0],"../vis/testNet_gridx_rem_heat.png")
    plotHeat(grid[0,:,:,1],"../vis/testNet_gridy_rem_heat.png")

    img3 = F.grid_sample(img1, grid)

    plotHeat(img3[0,0],"../vis/testNet_gridImg3.png")
    '''

    '''
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
    x += Variable(x.data.new(x.size()).normal_(0, 0.5))

    Image.fromarray(preprocc(x[0])).save("../vis/testNetBuilder_in.jpg")

    geom = GeomLayer(inChan,(200,200),inChan)

    opti = torch.optim.SGD(geom.parameters(), lr=0.05, momentum=0.9)

    for i in range(500):
        opti.zero_grad()

        finalPred,pred,mask = geom(x)

        loss = torch.pow(finalPred-target,2).mean()

        loss.backward()

        opti.step()

        if i % 50 == 0:
            print(i)
            Image.fromarray(preprocc(finalPred[0])).save("../vis/testNetBuilder_finalPred_{}.png".format(i))
            Image.fromarray(preprocc(pred[0])).save("../vis/testNetBuilder_pred_{}.png".format(i))
            Image.fromarray(preprocc(mask[0])).save("../vis/testNetBuilder_mask_{}.png".format(i))
    '''
    np.random.seed(0)
    torch.manual_seed(0)
    maxpool_g = MaxPool2d_G(torch.tensor([2,2]))

    size = 28

    rowInds = torch.arange(size).unsqueeze(1).expand(size,size)
    colInds = torch.arange(size).unsqueeze(0).expand(size,size)

    rowInds = rowInds.unsqueeze(0).unsqueeze(0).expand(1,1,size,size).float()
    colInds = colInds.unsqueeze(0).unsqueeze(0).expand(1,1,size,size).float()

    dist = torch.exp(-(torch.pow(rowInds - 2,2)+torch.pow(colInds - 2,2))/10)*7

    noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([0.0]), covariance_matrix=torch.eye(1)).sample(dist.size()).squeeze(dim=-1)

    dist = dist+noise

    dist = dist.expand(128,8,dist.size(-2),dist.size(-1))

    plotHeat(dist[0,0],"../vis/testMaxPool_in0.png")
    plotHeat(dist[0,1],"../vis/testMaxPool_in1.png")

    res = maxpool_g(dist)

    plotHeat(res[0,0],"../vis/testMaxPool_out0.png")
    plotHeat(res[0,0],"../vis/testMaxPool_out1.png")
