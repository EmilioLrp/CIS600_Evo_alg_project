import data_util
from mlp import MLPClassifier
import torch
from data_set import DataSet
import numpy as np

def calculate_loss(pred, label):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    loss = pred - label
    return loss

def train():
    train_x, test_x, train_y, test_y = data_util.load_data()
    train_data = DataSet(train_x, train_y)
    test_data = DataSet(test_x, test_y)
    epoch = 100
    batch_size = 100
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MLPClassifier()
    optimizer = torch.optim.Adam()
    loss_fn = torch.nn.CrossEntropyLoss(reduce='mean')
    for e in range(epoch):
        #TODO: finish this
        pass


if __name__ == "__main__":
    train()