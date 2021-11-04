import torch
from torch.utils.data import Dataset

class XOR_dataset(Dataset):

    def __init__(self,datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self,index):
        input_data = torch.tensor(self.datalist[index][0])
        label = torch.tensor(self.datalist[index][1])

        return input_data.cuda(), label.cuda()