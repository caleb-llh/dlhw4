import torch
import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
    def __init__(self, mode):
        super(Net, self).__init__()    
        if mode =='A':
            self.resnet18 = models.resnet18() # A
        else:
            self.resnet18 = models.resnet18(pretrained=True) # B
            if mode =='C':
                for params in list(self.resnet18.children())[:-2]: # C
                    params.requires_grad = False
        # self.linear = nn.Linear(in_features=1000, out_features=102)


    def forward(self, img):
        return self.resnet18(img)
        # return self.linear(self.resnet18(img))
