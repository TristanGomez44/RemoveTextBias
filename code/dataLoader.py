import  torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms
import os
from skimage import feature
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def loadData(dataset,batch_size,test_batch_size,permutate,cuda=False,num_workers=1,cropSize=150,trainProp=1):
    ''' Build two dataloader

    Args:
        dataset (string): the name of the dataset. Can be \'MNIST\' or \'CIFAR10\'.
        batch_size (int): the batch length for training
        test_batch_size (int): the batch length for testing
        perm (bool): Whether or not to permute the pixels of the image
        cuda (bool): whether or not to run computation on gpu
        num_workers (int): the number of workers for loading the data.
            Check pytorch documentation (torch.utils.data.DataLoader class) for more details
    Returns:
        train_loader (torch.utils.data.dataloader.DataLoader): the dataLoader for training
        test_loader (torch.utils.data.dataloader.DataLoader): the dataLoader for testing

    '''

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}

    dataDict = {"MNIST":(28*28,1),"FAKENIST":(28*28,1),"CIFAR10":(32*32,3),"IMAGENET":(cropSize*cropSize,3)}

    if permutate:
        permInd = np.arange(dataDict[dataset][0]*dataDict[dataset][1])
        np.random.shuffle(permInd)
    else:
        permInd = np.arange(dataDict[dataset][0]*dataDict[dataset][1])

    def permutFunc(x,permInd):
        origSize = x.size()
        x = x.view(-1)
        x = x[permInd]
        x = x.view(origSize)
        return x
    perm = transforms.Lambda(lambda x:permutFunc(x,permInd))

    if dataset == "MNIST":

        trainDataset = datasets.MNIST('../data/MNIST', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),perm]))

        subset_indices = torch.arange(int(len(trainDataset)*trainProp))

        train_loader = torch.utils.data.DataLoader(trainDataset,batch_size=batch_size,sampler=SubsetRandomSampler(subset_indices), **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/MNIST', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),perm])),
            batch_size=test_batch_size, shuffle=False, **kwargs)

    elif dataset == "CIFAR10":

        trainDataset = datasets.CIFAR10('../data/', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),perm]))
        subset_indices = torch.arange(int(len(trainDataset)*trainProp))

        train_loader = torch.utils.data.DataLoader(trainDataset,batch_size=batch_size,sampler=SubsetRandomSampler(subset_indices), **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),perm])),
            batch_size=test_batch_size, shuffle=False, **kwargs)

    elif dataset == "FAKENIST":

        dataset = datasets.FakeData(image_size=(1, 28, 28), num_classes=10,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),perm]))
        train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(dataset,batch_size=test_batch_size, shuffle=False, **kwargs)

    elif dataset == "IMAGENET":

        traindir = os.path.join("../data/ImageNet", 'train')
        valdir = os.path.join("../data/ImageNet", 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomResizedCrop(cropSize),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,perm]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=kwargs["num_workers"], pin_memory=True, sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256),transforms.CenterCrop(cropSize),transforms.ToTensor(),normalize,perm])),
            batch_size=test_batch_size, shuffle=False,
            num_workers=kwargs["num_workers"], pin_memory=True)

    else:
        raise ValueError("Unknown dataset",dataset)

    return train_loader,test_loader,permInd

if __name__ == '__main__':

    train,_ = loadData("CIFAR10",1,1,cuda=False,num_workers=1)

    print(type(train))
