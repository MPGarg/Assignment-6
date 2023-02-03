import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout=0.05, normalization='BN'):
        super(Net, self).__init__()
        
        ## Convolution Block1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias = False),  
            self.norm(normalization,32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, 3, padding=1, bias = False), 
            self.norm(normalization,64),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        
        ## Transition Block1
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 32,3, stride=2), 
            self.norm(normalization,32),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        ## Convolution Block2
        self.conv2 =  nn.Sequential(
            nn.Conv2d(32, 32, 3,  padding=1, bias = False), 
            self.norm(normalization,32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(32,32, 3,  padding=1,bias = False), 
            self.norm(normalization,32),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        
        #Transition Block2
        self.trans2 = nn.Sequential(

            nn.Conv2d(32, 16, 3, stride=2), 
            self.norm(normalization,16),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        #Convolution Block3
        self.conv3 = nn.Sequential(
            ## Depthwise Separable Convolution
            nn.Conv2d(16,16, 3,  padding=1,groups=16 ,bias = False),            
            nn.Conv2d(16, 32, 1,  padding=0, bias = False), 
            self.norm(normalization,32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            ## Dilation Block
            nn.Conv2d(32, 32, 3,  padding=1, bias = False,dilation=2), 
            self.norm(normalization,32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        #Transition Block3
        self.trans3 = nn.Sequential(

            nn.Conv2d(32, 16, 3, stride=2), 
            self.norm(normalization,16),
            nn.ReLU(),
            nn.Dropout2d(dropout)
            )

        #Convolution Block4        
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias = False), 
            self.norm(normalization,16),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 10, 3, padding=1, bias = False),         
            self.norm(normalization,10),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        ## Output Block
        self.out = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        ) 

    def norm(self,norm, channels):
        if norm == 'BN':
            return nn.BatchNorm2d(channels)
        elif norm == 'LN':
            return nn.GroupNorm(1,channels) #(equivalent with LayerNorm)
        elif norm == 'GN':
            return nn.GroupNorm(2,channels) #groups=2


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)

        x = self.conv2(x) 
        x = self.trans2(x) 

        x = self.conv3(x) 
        x = self.trans3(x)

        x = self.conv4(x)
        x = self.out(x)

        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1)




