import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pdb
import sys
import random
import pickle
from torch.utils.data import Dataset

from  model import *
from dataset import *

from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations

#### 1. LOAD pre_trained model
model = XOR_Net().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum = 0.9)

PATH = "./model.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
batch_size = checkpoint['batch_size']

pdb.set_trace()


#### 2. TEST
with open('data.pickle','rb') as f:
    data_list = pickle.load(f)

# dataset
XOR_dataset = XOR_dataset(data_list)
num_train = int(len(data_list)*0.9)
train_set, test_set = torch.utils.data.random_split(XOR_dataset, [num_train, len(data_list)-num_train])

# dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

# test - correct
model.eval()
corrects = 0
with torch.no_grad():
    for i, (input, target) in enumerate(tqdm(test_loader)):
        output = torch.round(model(input))
        target = torch.unsqueeze(target, 1).cuda()
        corrects += target.eq(output).sum().item()

print("correct {}/ total {}".format(corrects, len(test_loader)*batch_size))

# test - raw output
test_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], requires_grad=True)
targets = torch.tensor([0.,1.,1.,0.]).view(-1, 1)

for target, test_input in zip(targets,test_data):
    test_output = model(test_input.cuda())
    print("Input: {} | Output: {:4f} | Target: {}".format(test_input.data, test_output.item(), target.item()))

#### 3. PLOT model 3D graph
x = np.linspace(0,1,70)
y = np.linspace(0,1,70)
z = []

def f(x,y):
    input = torch.tensor([x,y]).float()
    z = model(input.cuda())
    # gpu -> cpu, GPU -> CPU
    z = z.detach().cpu().numpy()
    return z

all_points = [(x_point,y_point) for x_point in x for y_point in y]
for (x_point,y_point) in all_points:
    z.append(f(x_point,y_point))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for ((x,y),z) in zip(all_points, z):
    ax.scatter(x,y,z)
plt.show()