import  torch.utils.data
from torchvision import datasets, transforms
import os
def loadData(dataset,batch_size,test_batch_size,cuda=False,num_workers=1):
    ''' Build two dataloader

    Args:
        dataset (string): the name of the dataset. Can be \'MNIST\' or \'CIFAR10\'.
        batch_size (int): the batch length for training
        test_batch_size (int): the batch length for testing
        cuda (bool): whether or not to run computation on gpu
        num_workers (int): the number of workers for loading the data.
            Check pytorch documentation (torch.utils.data.DataLoader class) for more details
    Returns:
        train_loader (torch.utils.data.dataloader.DataLoader): the dataLoader for training
        test_loader (torch.utils.data.dataloader.DataLoader): the dataLoader for testing

    '''

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}

    if dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/MNIST', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/MNIST', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=test_batch_size, shuffle=False, **kwargs)

    elif dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=test_batch_size, shuffle=False, **kwargs)
    elif dataset == "FAKE":
        train_loader = torch.utils.data.DataLoader(datasets.FakeData(image_size=(3, 28, 28),transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.FakeData(image_size=(3, 28, 28),transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=test_batch_size, shuffle=False, **kwargs)

    elif dataset == "FAKENET":

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(datasets.FakeData(image_size=(3, 256, 256),transform=transforms.Compose([
                               transforms.ToTensor(),normalize])),batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.FakeData(image_size=(3, 256, 256),transform=transforms.Compose([
                               transforms.ToTensor(),normalize])),batch_size=test_batch_size, shuffle=False, **kwargs)
    elif dataset == "FAKENIST":

        normalize = transforms.Normalize(mean=[0.1307],std=[0.3081])

        train_loader = torch.utils.data.DataLoader(datasets.FakeData(image_size=(1, 100, 100),transform=transforms.Compose([
                               transforms.ToTensor(),normalize])),batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.FakeData(image_size=(1, 100, 100),transform=transforms.Compose([
                               transforms.ToTensor(),normalize])),batch_size=test_batch_size, shuffle=False, **kwargs)




    elif dataset == "IMAGENET":

                # Data loading code
        traindir = os.path.join("../data/ImageNet", 'train')
        valdir = os.path.join("../data/ImageNet", 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=kwargs["num_workers"], pin_memory=True, sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=test_batch_size, shuffle=False,
            num_workers=kwargs["num_workers"], pin_memory=True)

    else:
        raise ValueError("Unknown dataset",dataset)

    return train_loader,test_loader

if __name__ == '__main__':

    train,_ = loadData("CIFAR10",1,1,cuda=False,num_workers=1)

    print(type(train))
