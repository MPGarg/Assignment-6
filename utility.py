#!pip install -U albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def set_albumen_params(mean, std):
    horizontalflip_prob= 0.2
    rotate_limit= 15
    shiftscalerotate_prob= 0.25
    num_holes= 1
    cutout_prob= 0.5
    max_height = 16 #32/2
    max_width = 16 #32/2

    transform_train = A.Compose(
      [A.HorizontalFlip(p=horizontalflip_prob),
      A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=rotate_limit, p=shiftscalerotate_prob),
      A.CoarseDropout(max_holes=num_holes,min_holes = 1, max_height=max_height, max_width=max_width, 
      p=cutout_prob,fill_value=tuple([x * 255.0 for x in mean]),
      min_height=max_height, min_width=max_width, mask_fill_value = None),
      A.Normalize(mean = mean, std = std, max_pixel_value=255, always_apply = True),
      ToTensorV2()
      ])
    
    transform_valid = A.Compose(
      [
      A.Normalize(
              mean=mean,
              std=std,
              max_pixel_value=255,
          ),
      ToTensorV2()
      ])
    return transform_train, transform_valid 

def load_data():
    transform = transforms.Compose(
      [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
    return trainloader, trainset    

def display_incorrect_pred(mismatch, n=20 ):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    display_images = mismatch[:n]
    index = 0
    fig = plt.figure(figsize=(20,20))
    for img in display_images:
        image = img[0].squeeze().to('cpu').numpy()
        pred = classes[img[1]]
        actual = classes[img[2]]
        ax = fig.add_subplot(4, 5, index+1)
        ax.axis('off')
        ax.set_title(f'\n Predicted Label : {pred} \n Actual Label : {actual}',fontsize=10) 
        ax.imshow(np.transpose(image, (1, 2, 0))) 
        #ax.imshow(image, cmap='gray_r')
        index = index + 1
    plt.show()