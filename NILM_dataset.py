import torch
import torch.utils.data
import numpy as np
import sys
#sys.path.append('../IpqLS')

#from dataset_utils import *

class NILMDataset(torch.utils.data.Dataset):
    def __init__(self, x, house="house_5"):

        self.x = self.normalize(x)

        #print(np.max(self.p))
        #print(np.min(self.p))
        #print(np.max(self.q))
        #print(np.min(self.q))
        
    def normalize(self, x):
        mean = np.mean(x)
        
        std  = np.std(x)

        x = (x - mean) / std

        return x

    def __getitem__(self, index):
        return self.x[index]
        
    def __len__(self):
        return len(self.x)
