import sys
from args import ArgReader
from args import str2bool
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from torch.autograd import Variable
import netBuilder
import os
import dataLoader
import configparser
import torch.nn.functional as F
import vis
from torch.distributions import Bernoulli
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from PIL import Image

def trainDetect(model,optimizer,train_loader, epoch, writer,args):
    '''Train a detecting network

    After having run the net on every image of the train set,
    its state is saved in the nets/NameOfTheExperience/ folder

    Args:
        model (CNN): a CNN module (as defined in netBuilder) with two outputs
        optimizer (torch.optim): the optimizer to train the network
        train_loader (torch.utils.data.DataLoader): the loader to generate batches of train images
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the network

    '''

    model.train()
    print("Train epoch : ",epoch)
    total_loss = 0
    total_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        output,_ = model(data)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(target.data.view_as(pred)).float().cpu().sum()

        total_acc += correct*1.0/len(pred)

        loss = F.cross_entropy(output, target)
        loss.backward()

        total_loss += loss.data.item()

        optimizer.step()

        if batch_idx > 3 and args.debug:
            break

    writeSummaries(total_loss,total_acc,batch_idx+1,writer,epoch,"train",args.model_id,args.exp_id)

    torch.save(model.state_dict(), "../nets/{}/model{}_epoch{}".format(args.exp_id,args.model_id, epoch))

def testDetect(model,test_loader,epoch, writer,args):
    '''Test a detecting network
    Compute the accuracy and the loss on the test set and write every output score of the net in a csv file

    Args:
        model (CNN): a CNN module (as defined in netBuilder) with two outputs
        test_loader (torch.utils.data.DataLoader): the loader to generate batches of test images
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the network

    '''

    model.eval()
    print("Test epoch : ",epoch)
    total_loss = 0
    total_acc = 0

    #The header of the csv file is written after the first test batch
    firstTestBatch = True

    for batch_idx,(data, target) in enumerate(test_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output,_ = model(data)

        total_loss += F.cross_entropy(output, target).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability


        total_acc += pred.eq(target.data.view_as(pred)).float().cpu().sum()/len(pred)

        firstTestBatch = False

    #Print the results
    writeSummaries(total_loss,total_acc,batch_idx+1,writer,epoch,"val",args.model_id,args.exp_id)

def writeSummaries(total_loss,total_acc,sampleNb,writer,epoch,mode,model_id,exp_id):

    total_loss /= sampleNb
    total_acc /= sampleNb

    writer.add_scalars('Losses',{model_id+"_"+mode:total_loss},epoch)
    writer.add_scalars('Accuracies',{model_id+"_"+mode:total_acc},epoch)

    if not os.path.exists("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode)):
        header = "epoch,loss,acc"
    else:
        header = ""

    with open("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode),"a") as text_file:
        print(header,file=text_file)
        print("{},{},{}\n".format(epoch,total_loss,total_acc),file=text_file)

def findLastNumbers(weightFileName):
    '''Extract the epoch number of a weith file name.

    Extract the epoch number in a weight file which name will be like : "clustDetectNet2_epoch45".
    If this string if fed in this function, it will return the integer 45.

    Args:
        weightFileName (string): the weight file name
    Returns: the epoch number

    '''

    i=0
    res = ""
    allSeqFound = False
    while i<len(weightFileName) and not allSeqFound:
        if not weightFileName[len(weightFileName)-i-1].isdigit():
            allSeqFound = True
        else:
            res += weightFileName[len(weightFileName)-i-1]
        i+=1

    res = res[::-1]

    return int(res)

def get_OptimConstructor_And_Kwargs(optimStr,momentum):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr != "AMSGrad":
        optimConst = getattr(optim,optimStr)
        if optimStr == "SGD":
            kwargs= {'momentum': momentum}
        elif optimStr == "Adam":
            kwargs = {}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = optim.Adam
        kwargs = {'amsgrad':True}

    print("Optim is :",optimConst)

    return optimConst,kwargs

def initialize_Net_And_EpochNumber(net,start_mode, init,exp_id,model_id,cuda,netType):
    '''Initialize a clustering detecting network

    Can initialise with parameters from detecting network or from a clustering detecting network

    If init is None, the network will be left unmodified. Its initial parameters will be saved.

    Args:
        net (CNN): the net to be initialised
        pretrain (boolean): if true, the net trained is a detectNet (can be used after to initialize the detectNets of a clustDetectNet)
        init (string): the path to the weigth for initializing the net with
        exp_id (string): the name of the experience
        model_id (int): the id of the network
        cuda (bool): whether to use cuda or not

    Returns: the start epoch number
    '''

    #Initialize the detect nets with weights from a supervised training
    if start_mode == "fine_tune":
        params = torch.load(init)

        net.load_state_dict(params)
        startEpoch = findLastNumbers(init)+1

    #Starting a network from scratch
    elif start_mode == "scratch":
        #Saving initial parameters
        torch.save(net.state_dict(), "../nets/{}/model{}_epoch0".format(exp_id,model_id))
        startEpoch = 1
    else:
        raise ValueError("Unknown start mode : ",start_mode)

    return startEpoch

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--noise', type=float, metavar='NOISE',
                        help='the amount of noise to add in the gradient of the clustNet (in percentage)(default: 0.1)')

    argreader.parser.add_argument('--optim', type=str, default="SGD", metavar='OPTIM',
                        help='the optimizer algorithm to use (default: \'SGD\')')
    argreader.parser.add_argument('--init', type=str,  default=None,metavar='N', help='the weights to use to initialize the detectNets')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader,test_loader,perm = dataLoader.loadData(args.dataset,args.batch_size,args.test_batch_size,args.permutate,args.cuda,args.num_workers)

    if args.write_img_ex:

        for i in range(10):
            tensor = test_loader.dataset[i][0]
            vis.writeImg('../vis/{}/{}_img{}.jpg'.format(args.exp_id,args.dataset,i),tensor.detach().numpy())

            origSize = tensor.size()
            tensor = tensor.view(-1)
            tensor = tensor[np.argsort(perm)]
            tensor = tensor.view(origSize)

            vis.writeImg('../vis/{}/{}_img{}_noperm.jpg'.format(args.exp_id,args.dataset,i),tensor.detach().numpy())

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../nets/{}".format(args.exp_id))):
        os.makedirs("../nets/{}".format(args.exp_id))

    netType = "net"

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../nets/{}/{}{}.ini".format(args.exp_id,netType,args.model_id))

    #The writer for tensorboardX
    writer = SummaryWriter("../results/{}".format(args.exp_id))

    print("Model :",args.model_id,"Experience :",args.exp_id)

    #Building the net
    net = netBuilder.netMaker(args)

    if args.cuda:
        net.cuda()

    startEpoch = initialize_Net_And_EpochNumber(net,args.start_mode,args.init_path,args.exp_id,args.model_id,args.cuda,netType)

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargs = get_OptimConstructor_And_Kwargs(args.optim,args.momentum)

    #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
    #the args.lr argument will be a float and not a float list.
    #Converting it to a list with one element makes the rest of processing easier
    if type(args.lr) is float:
        args.lr = [args.lr]

    #Train and evaluate the clustering detecting network for several epochs
    lrCounter = 0

    for epoch in range(startEpoch, args.epochs + 1):

        #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
        #The optimiser have to be rebuilt every time the learning rate is updated
        if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==startEpoch:

            kwargs['lr'] = args.lr[lrCounter]
            print("Learning rate : ",kwargs['lr'])
            optimizer = optimConst(net.parameters(), **kwargs)

            if lrCounter<len(args.lr)-1:
                lrCounter += 1

        trainDetect(net,optimizer,train_loader,epoch, writer,args)
        testDetect(net,test_loader,epoch, writer,args)

if __name__ == "__main__":
    main()
