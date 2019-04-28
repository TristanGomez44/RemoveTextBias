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

class Triang(nn.Module):
    def __init__(self):
        super(Triang,self).__init__()
    def forward(self,x):
        return (1+x)*((-1<x)*(x<=0)).float() + (1-x)*((0<x)*(x<=1)).float()

class Relu(nn.Module):
    def __init__(self):
        super(Relu,self).__init__()
    def forward(self,x):
        return F.relu(x)


class BasicConv2d(nn.Module):
    """A basic 2D convolution layer

    This layer integrates 2D batch normalisation and relu activation

    Comes mainly from torchvision code :
    https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    Consulted : 19/11/2018

    """
    def __init__(self, in_channels, out_channels,use_bn=True,activation=Relu, **kwargs):

        '''
        Args:
            in_channels (int): the number of input channel
            out_channels (int): the number of output channel
            use_bn (boolean): whether or not to use 2D-batch normalisation
             **kwargs: other argument passed to the nn.Conv2D constructor
        '''

        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not(use_bn), **kwargs)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)

        return self.activation(x)

class ConvFeatExtractor(nn.Module):
    """A convolutional feature extractor

    It is a stack of BasicConv2d layers with residual connections and it also comprise
    two maxpooling operation (one after layer 1 and one after the last layer)

    """

    def __init__(self,inChan,chan,nbLay,ker,maxPl1,maxPl2,applyDropout2D,outChan=None,activation=Relu):
        '''
        Args:
            inChan (int): the number of input channel
            chan (int): the number of channel in every layer of the network
            outChan (int): the number of channel at the end layer of the network. Default value is the number of channel\
                in the other layers.
            avPool (boolean): whether of not to use average pooling
            nbLay (int): the number of layer of the net
            ker (int): the size of side of the kernel to use (the kernel is square)
            maxPl1 (int): the max-pooling size after the first convolutional layer
            maxPl2 (int): the max-pooling size after the before-to-last convolutional layer
            applyDropout2D (boolean): whether or not to use 2D-dropout on the middle layers during training
        '''
        super(ConvFeatExtractor,self).__init__()

        if outChan is None:
            self.outChan = chan
        else:
            self.outChan = outChan

        self.nbLay = nbLay
        self.chan = chan
        self.inChan = inChan
        self.ker = ker

        self.maxPl1 = maxPl1
        self.maxPl2 = maxPl2

        self.poollLay1 = nn.MaxPool2d(maxPl1,return_indices=True)
        self.poollLay2 = nn.MaxPool2d(maxPl2,return_indices=True)

        self.applyDropout2D = applyDropout2D

        self.convs = nn.ModuleList([BasicConv2d(inChan, chan,kernel_size=self.ker,activation=activation)])

        self.padd = nn.ZeroPad2d(self.computePaddingSize(self.ker))

        if self.nbLay > 2:

            #Padding is applied on every layer so that the feature map size stays the same at every layer
            self.convs.extend([BasicConv2d(chan,chan,kernel_size=self.ker,activation=activation) for i in range(self.nbLay-2)])

        self.convs.append(BasicConv2d(chan, outChan,kernel_size=self.ker,activation=activation))

        self.drop2DLayer = nn.Dropout2d()

    def forward(self,x):
        ''' Compute the forward pass of the stacked layer
        Returns:
            x (torch.autograd.variable.Variable): the processed batch
            actArr (list) the list of activations array of each layer
            maxPoolInds (dict): a dictionnary containing two objects : the indexs of the maximum elements for the first maxpooling
                and the second maxpooling. These two objects are obtained by the return of the nn.MaxPool2d() function
        '''

        actArr = []
        netshape ="in : "+str(x.size())+"\n"
        maxPoolInds = {}

        for i, l in enumerate(self.convs):
            if  i != 0 and self.applyDropout2D:
                x = self.drop2DLayer(x)

            #Compute next layer activations
            if (i != len(self.convs)-1 or self.chan==self.outChan) and (i != 0 or self.chan==self.inChan):
                #Add residual connection for all layer except the first and last (because channel numbers need to match)
                x = self.padd(l(x))+x
            else:
                x = self.padd(l(x))

            actArr.append(x)
            netshape +="conv : "+str(x.size())+"\n"

            if i == len(self.convs)//2-1:
                x,inds1 = self.poollLay1(x)
                maxPoolInds.update({"inds1":inds1})
            elif i == len(self.convs)-1:

                x,inds2 = self.poollLay2(x)
                maxPoolInds.update({"inds2":inds2})

        actArr.append(x)
        return x,actArr,maxPoolInds

    def computePaddingSize(self,kerSize):
        ''' Compute the padding size necessary to compensate the size reduction induced by a conv operation
        Args:
            kerSize (int): the size of the kernel (assumed squarred)
        '''

        halfPadd = (kerSize-1)//2

        if kerSize%2==0:
            padd = (halfPadd,1+halfPadd,halfPadd,1+halfPadd)
        else:
            padd = (halfPadd,halfPadd,halfPadd,halfPadd)

        return padd

class CNN(nn.Module):
    """A CNN module"""

    def __init__(self,inSize,inChan,chan,avPool,nbLay,ker,maxPl1,maxPl2,applyDropout2D,nbDenseLay,sizeDenseLay,nbOut,applyLogSoftmax=True,\
                outChan=None,convActivation='ReLU',denseActivation='ReLU'):
        """
        Args:
            nbOut (int): the number of output at the last dense layer
            applyLogSoftmax (bool): whether or not to apply the nn.functional.log_softmax function in the last dense layer
            other arguments : check ConvFeatExtractor module constructor
        """

        if outChan is None:
            outChan=chan

        super(CNN,self).__init__()

        if convActivation == "ReLU":
            convActivationFunc = Relu()
        elif convActivation == "triang":
            convActivationFunc = Triang()
        else:
            raise ValueError("Unknown activation function for the conv layers : {}".format(convActivation))

        self.avPool = avPool
        if nbLay != 0:
            self.convFeat = ConvFeatExtractor(inChan=inChan,chan=chan,outChan=outChan,nbLay=nbLay,\
                                              ker=ker,maxPl1=maxPl1,maxPl2=maxPl2,applyDropout2D=applyDropout2D,activation=convActivationFunc)
        else:
            self.convFeat = None

        self.applyLogSoftmax = applyLogSoftmax

        if denseActivation == "ReLU":
            self.activation = Relu
        elif denseActivation == "triang":
            self.activation = Triang
        else:
            raise ValueError("Unknown activation function for the dense layers : {}".format(denseActivation))

        if nbDenseLay == 1:

            if nbLay != 0:
                if not self.avPool:
                    InputConv2Size = inSize//maxPl1
                    InputLinearSize = InputConv2Size//maxPl2
                    dense = nn.Linear(InputLinearSize*InputLinearSize*outChan,nbOut)
                else:
                    dense = nn.Linear(outChan,nbOut)
            else:
                #If the input has 0 channel it indicates it is a vector and not an matrix representation of the input
                if inChan==0:
                    dense = nn.Linear(inSize,nbOut)
                else:
                    dense = nn.Linear(inSize*inSize*inChan,nbOut)

            self.denseLayers = dense

        elif nbDenseLay == 0:
            self.denseLayers = None

        else:

            if self.avPool:
                raise ValueError("Cannot use average pooling and more than one dense layer")
            print("More than one dense layer")

            if nbLay != 0:
                InputConv2Size = int((inSize-(ker-1))/maxPl1)
                InputLinearSize = int((InputConv2Size-(ker-1))/maxPl2)

                self.denseLayers = nn.ModuleList([nn.Linear(InputLinearSize*InputLinearSize*outChan,sizeDenseLay),self.activation])
            else:

                InputLinearSize = inSize
                #If the input has 0 channel it indicates it is a vector and not an matrix representation of the input

                if inChan==0:
                    self.denseLayers = nn.ModuleList([nn.Linear(InputLinearSize,sizeDenseLay),self.activation])
                else:
                    self.denseLayers = nn.ModuleList([nn.Linear(InputLinearSize*InputLinearSize*inChan,sizeDenseLay),self.activation])

            for i in range(nbDenseLay-2):
                self.denseLayers.extend([nn.Linear(sizeDenseLay,sizeDenseLay),self.activation])

            self.denseLayers.append(nn.Linear(sizeDenseLay,nbOut))
            self.denseLayers = nn.Sequential(*self.denseLayers)


    def forward(self,x):
        '''Computes the output of the CNN

        Returns:
            x (torch.autograd.variable.Variable): the batch of predictions
            actArr (the list of activations array of each layer)
        '''

        actArr = []

        if self.convFeat:
            x,actArr,netShape = self.convFeat(x)

        if self.avPool:
            x = x.sum(dim=-1,keepdim=True).sum(dim=-2,keepdim=True)

        if self.denseLayers:

            #Flattening the convolutional features
            x = x.view(x.size()[0],-1)

            x = self.denseLayers(x)
            actArr.append(x)

            if self.applyLogSoftmax:
                x = F.log_softmax(x, dim=1)

        return x,actArr

    def setWeights(self,params,cuda,noise_init):
        '''Set the weight of the extractor
        Args:
            params (dict): the dictionnary of tensors used to set the extractor parameters. This must be the parameters of a CNN module
            cuda (bool): whether or not to use cuda
            noise_init (float): the proportion of noise to add to the weights (relative to their norm). The noise is sampled from
            a Normal distribution and then multiplied by the norm of the tensor times half this coefficient. This means that
            95%% of the sampled noise will have its norm value under noise_init percent of the parameter tensor norm value

        '''
        #Used to determine if a parameter tensor has already been set.
        setKeys = []

        for key in params.keys():
            #The parameters may come from a CNN or CAE module so the keys might start with "convFeat" or "convDec"
            #newKey = key.replace("convFeat.","").replace("convDec.","")
            newKey = key

            #Adding the noise
            if noise_init != 0:

                noise = torch.randn(params[key].size())

                if cuda:
                    noise = noise.cuda()

                params[key] += noise_init*0.5*torch.pow(params[key],2).sum()*noise

            if cuda:
                params[key] = params[key].cuda()

            if newKey in self.state_dict().keys():
                if newKey.find("dense") == -1:
                    self.state_dict()[newKey].data += params[key].data -self.state_dict()[newKey].data
            #else:
            #    print("Cannot find parameters {}".format(newKey))

class ResNetBased(nn.Module):

    def __init__(self,resnet,resnetLayNb,layCut):

        super(ResNetBased,self).__init__()
        self.resnet = resnet

        if resnetLayNb == 50:
            baseFeatNb = 256
        elif resnetLayNb == 18:
            baseFeatNb = 64

        nbFeat = baseFeatNb*2**(layCut-1)

        self.dense = nn.Linear(nbFeat,10)

    def forward(self,x):

        x = self.resnet(x)
        x = self.dense(x)

        return x

def netMaker(args):
    '''Build a network
    Args:
        args (Namespace): the namespace containing all the arguments required for training and building the network
    Returns:
        the built network
    '''

    #Setting the size and the number of channel depending on the dataset
    if args.dataset == "MNIST":
        inSize = 28
        inChan = 1
        nbOut = 10
    elif args.dataset == "CIFAR10":
        inSize = 32
        inChan = 3
        nbOut = 10
    elif args.dataset == "FAKE":
        inSize = 28
        inChan = 3
        nbOut = 10
    elif args.dataset == "FAKENET":
        inSize = 256
        inChan = 3
        nbOut = 1000
    elif args.dataset == "FAKENIST":
        inSize = 100
        inChan = 1
        nbOut = 10
    elif args.dataset == "IMAGENET":
        inSize = 224
        inChan = 3
        nbOut = 1000
    else:
        raise("netMaker: Unknown Dataset")

    if args.modeltype=="cnn":

        net = CNN(inSize=inSize,inChan=inChan,chan=args.dechan,avPool=args.avpool,nbLay=args.denblayers,\
              ker=args.deker,maxPl1=args.demaxpoolsize,maxPl2=args.demaxpoolsize_out,applyDropout2D=args.dedrop,nbOut=nbOut,\
              applyLogSoftmax=False,nbDenseLay=args.denb_denselayers,sizeDenseLay=args.desize_denselayers,denseActivation=args.dense_activation,convActivation=args.conv_activation)
    else:
        net = resnet.resnet18(pretrained=True,layCut=4,activation=args.conv_activation)

    return net
