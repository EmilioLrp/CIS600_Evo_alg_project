import torch
import numpy as np

"""
reference: 
https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2
"""
class DataSet(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train):
        # self.X = torch.from_numpy(np.array(X_train).astype(np.float32))
        # self.y = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
        # self.y = torch.from_numpy(np.array(y_train).astype(np.float32))
        self.X = torch.FloatTensor(np.array(X_train))
        self.y = torch.FloatTensor(np.array(y_train))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len