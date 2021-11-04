import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import pdb
import sys
import random
import  copy
import pickle
from torch.utils.data import Dataset

from  model import *
from dataset import *


###################################################################
#                        Hyperparameters 
###################################################################
# SET seeds
random_seed = 5028
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# hyperparameters
learning_rate = 0.05
epochs = 7
batch_size = 4

# device configuration
os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
device = torch.device("cuda") # in case using .to(device)



####################################################################
#                          LOAD data
####################################################################
with open('data.pickle','rb') as f:
    data_list = pickle.load(f)

# dataset
XOR_dataset = XOR_dataset(data_list)
num_train = int(len(data_list)*0.9)
train_set, test_set = torch.utils.data.random_split(XOR_dataset, [num_train, len(data_list)-num_train])

# dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)



####################################################################
#                         DEFINE Model
####################################################################
# XOR Net
XOR_Net = XOR_Net().cuda()
print(next(XOR_Net.parameters()).is_cuda) # returns a boolean

# optimizer and criterion
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(XOR_Net.parameters(), lr = learning_rate, momentum = 0.9)



####################################################################
#                         TRAIN Model
####################################################################
for idx in range(epochs):
    for i, (input, target) in enumerate(train_loader):
        # forward propagation
        output = XOR_Net(input).cuda()
        target = torch.unsqueeze(target, 1).cuda()
        
        # Get Loss, Compute Gradient, Update Parameters
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print Loss for Tracking Training
        if i%200 == 0:
            print("Epoch: {: >8} | Steps: {: >8} | Loss: {:8f}".format(idx, i, loss))




####################################################################
#                         SAVE Model
####################################################################
EPOCH = idx
LOSS = loss
PATH = "./model.pt"
BATCH_SIZE = batch_size
# save checkpoint
torch.save({
            'epoch': EPOCH,
            'model_state_dict': XOR_Net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            'batch_size': BATCH_SIZE
            }, PATH)