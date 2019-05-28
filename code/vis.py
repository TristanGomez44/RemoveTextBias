"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import numpy as np
import sys
import torch

from torchvision import models
from torchvision import datasets, transforms

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import netBuilder
from skimage.transform import resize
from matplotlib.mlab import PCA
import dataLoader
from args import ArgReader

import random

#Look for the most important pixels in the activation value using mask
def salMap_mask(image,model,imgInd, maskSize=(1,1)):
    print("Computing saliency map using mask")
    pred,_ = model(image)

    argMaxpred = np.argmax(pred[0].detach().numpy())

    salMap = np.zeros((image.size()[2],image.size()[3]))

    for i in range(int(image.size()[2]/maskSize[0])):
        for j in range(int(image.size()[3]/maskSize[1])):

            maskedImage = torch.tensor(image)


            xPosS = i*maskSize[0]
            yPosS = j*maskSize[1]

            xPosE = min(image.size()[2],xPosS+maskSize[0])
            yPosE = min(image.size()[3],yPosS+maskSize[1])

            """
            print("------ New mask ------")
            print(maskedImage[0][0][xPosS:xPosE])
            print(yPosS)
            print(maskedImage[0][0][xPosS:xPosE,yPosS])
            """
            maskedImage[0][0][xPosS:xPosE,yPosS:yPosE] = image.min()
            maskedPred,_ = model(maskedImage)

            err = torch.pow((pred[0][argMaxpred] - maskedPred[0][argMaxpred]),2)
            salMap[xPosS:xPosE,yPosS:yPosE] = err.detach().numpy()

    writeImg("../vis/salMapMask_img_{}_u{}.png".format(imgInd,argMaxpred),salMap)

#Look for the most important pixels in the activation value using derivative
def salMap_der(image,model,imgInd):
    print("Computing saliency map using derivative")
    pred,_ = model(image)

    argMaxpred = np.argmax(pred.detach().numpy())

    loss = - pred[0][argMaxpred]

    loss.backward()

    salMap = image.grad/image.grad.sum()

    writeImg("../vis/salMapDer_img_{}_u{}.png".format(imgInd,argMaxpred),salMap.numpy()[0,0])

def opt(image,model,exp_id,model_id,imgInd, unitInd, epoch=1000, nbPrint=20, alpha=6, beta=2,
        C=20, B=2, stopThre = 0.000005,lr=0.001,momentum=0.9,optimType="SGD",layToOpti="conv",reg_weight=0):
    print("Maximizing activation")

    model.eval()
    Bp = 2*B
    V=B/2

    if optimType=="SGD":
        optimizer = optim.SGD([image], lr=lr,momentum=momentum,nesterov=True)
    else:
        optimizer = optim.LBFGS([image],lr=lr)

    i=0
    lastVar = stopThre
    last_img = np.copy(image.detach().numpy())

    while i<epoch and lastVar >= stopThre:

    #for i in range(1, epoch+1):
        optimizer.zero_grad()

        output,actArr = model(image)

        if layToOpti == "conv":
            act = actArr[-2][0,unitInd].mean()
        elif layToOpti == "logit":
            act = output[0,unitInd]

        # Loss function is minus the mean of the output of the selected layer/filter

        # computing the norm : +infinity if one pixel is above the limit,
        # else, computing a soft-constraint, the alpha norm (raised to the alpha power)
        if image.detach().numpy()[0,:].any() > Bp:
            norm = torch.tensor(float("inf"))
        else:
            norm = torch.sum(torch.pow(image,alpha))/float(image.size()[2]*image.size()[3]*np.power(B,alpha))

        # computing TV
        h_x = image.size()[2]
        w_x = image.size()[3]
        h_tv = torch.pow((image[:,:,1:,1:]-image[:,:,:h_x-1,:w_x-1]),2)
        w_tv = torch.pow((image[:,:,1:,1:]-image[:,:,:h_x-1,:w_x-1]),2)
        tv =  torch.pow(h_tv+w_tv,beta/2).sum()/(h_x*w_x*np.power(V,beta))

        loss = -C*act+reg_weight*(norm+tv)

        # Backward
        loss.backward(retain_graph=True)

        if optimType== 'LBFGS':
            def closure():
                optimizer.zero_grad()
                output = model(image)
                loss = -C*act+reg_weight*(norm+tv)
                loss.backward(retain_graph=True)
                return loss

            # Update image
            optimizer.step(closure)
        else:
            # Update image
            optimizer.step()

        np_img = np.copy(image.detach().numpy())
        lastVar = np.sqrt(np.power(last_img - np_img,2).sum())
        last_img = np.copy(image.detach().numpy())


        if i % nbPrint == 0:
            # Save image
            print('Iteration:', str(i), 'Loss:', loss.data.numpy(),"Travelled distance",lastVar)
            writeImg('../vis/{}/img{}_model{}_'.format(exp_id,imgInd,model_id)+'_u' + str(unitInd) + '_iter'+str(i)+'.jpg',image.detach().numpy()[0,:])

        i += 1
        #print(i,epoch,i<epoch,lastVar,stopThre,lastVar >= stopThre)

    writeImg('../vis/{}/img{}_model{}_'.format(exp_id,imgInd,model_id)+'_u' + str(unitInd) + '_iter'+str(i)+'.jpg',image.detach().numpy()[0,:])

def writeImg(path,img):

    np_img = resize(img,(img.shape[0],300,300),mode="constant", order=0,anti_aliasing=True)

    np_img = (np_img-np_img.min())/(np_img.max()-np_img.min())
    np_img = (255*np_img).astype('int')

    np_img = np.transpose(np_img, (1, 2, 0))

    cv2.imwrite(path,np_img)

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--max_act', type=str,nargs=4, metavar='NOISE',
                        help='To visualise an image that maximise the activation of one unit in the last layer. \
                        The values are :\
                            the path to the model, \
                            the number of image to be created, \
                            the layer to optimise. Can be \'conv\' or \'dense\' \
                            the unit to optimise. If not indicated, the unit number i will be optimised if image has label number i.')

    argreader.parser.add_argument('--stop_thres', type=float, default=0.000005,metavar='NOISE',
                        help='If the distance travelled by parameters during activation maximisation become lesser than this parameter, the optimisation stops.')

    argreader.parser.add_argument('--reg_weight', type=float, default=0,metavar='NOISE',
                        help='The weight of the regularisation during activation maximisation.')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))

    if args.max_act:

        modelPath = args.max_act[0]
        nbImages = int(args.max_act[1])
        layToOpti = args.max_act[2]
        unitInd = int(args.max_act[3])

        random.seed(args.seed)

        #Building the net
        model = netBuilder.netMaker(args)
        model.load_state_dict(torch.load(modelPath))

        _,test_loader = dataLoader.loadData(args.dataset,args.batch_size,1,args.cuda,args.num_workers)

        #Comouting image that maximises activation of the given unit in the given layer
        maxInd = len(test_loader.dataset) - 1

        model.eval()

        for i,(image,label) in enumerate(test_loader):

            print("Image ",i)

            img = Variable(test_loader.dataset[i][0])

            #if img.size(1) != 3:
            #    img = img.expand((img.size(0),3,img.size(2),img.size(3)))

            writeImg('../vis/{}/img_'.format(args.exp_id)+str(i)+'.jpg',image.detach().numpy())

            img.requires_grad = True

            #if unitInd is None:
            #    unitInd = label.item()

            opt(img,model,args.exp_id,args.model_id,i,unitInd=unitInd,lr=args.lr,momentum=args.momentum,optimType='LBFGS',layToOpti=layToOpti,\
                epoch=args.epochs,nbPrint=args.log_interval,stopThre=args.stop_thres,reg_weight=args.reg_weight)

            #salMap_der(img,model,i)
            #salMap_mask(img,model,i)

            if i == nbImages-1:
                break


if __name__ == "__main__":
    main()
