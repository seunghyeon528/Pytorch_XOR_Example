import torch
import torch.nn as nn
import torch.nn.functional as F



class XOR_Net(nn.Module):
    def __init__(self):
        super(XOR_Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
        
