from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision

class cifar_ds10(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def tl_ts_mod(transform_train,transform_valid):
    trainset = cifar_ds10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = cifar_ds10(root='./data', train=False, download=True, transform=transform_valid)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    return trainset,trainloader,testset,testloader