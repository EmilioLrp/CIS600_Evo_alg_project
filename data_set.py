import torch
import numpy as np

"""
reference: 
https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2
"""
class DataSet(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.FloatTensor(np.array(X_train))
        self.y = torch.FloatTensor(y_train)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len