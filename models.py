import torch
import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
    def __init__(self, mode):
        super(Net, self).__init__()    
        if mode =='A':
            self.resnet18 = models.resnet18() # A
        else:
            self.resnet18 = models.resnet18(pretrained=True) # B & C
        self.resnet18.fc = torch.nn.Linear(512, 102)
        if mode =='C':
            for params in list(self.resnet18.children())[:-2]: # C
                params.requires_grad = False
    

    def forward(self, img):
        return self.resnet18(img)

if __name__ == "__main__":
    B = Net('B')
    C = nn.Sequential(*list(models.resnet18().children())[:-2])
    print(B)
    # print(C)