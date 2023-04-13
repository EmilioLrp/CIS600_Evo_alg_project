from data_set import DataSet
from mlp import MLPClassifier
import torch
import data_util
import numpy as np
import os 
import pickle as pk

def calculate_train_acc(pred, label):
    pred = pred.cpu().detach()
    pred = torch.nn.functional.softmax(pred).numpy()
    max_idx = np.argmax(pred, -1)
    pred_round = np.zeros(pred.shape)
    for i in range(pred.shape[0]):
        pred_round[i, max_idx[i]] = float(1)
    # pred = np.round(pred)
    label = label.cpu().detach().numpy()
    # acc = float(np.sum(pred_round == label))/label.shape[0]
    acc = 0
    for i in range(label.shape[0]):
        if np.array_equal(pred_round[i, :], label[i, :]):
            acc += 1
    return float(acc)/ float(label.shape[0])

def calculate_test_acc(model, testing_data):
    batch_size = 100
    testloader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True, num_workers=2)
    correct = 0
    count = 0
    for i, data in enumerate(testloader):
        input, label = data
        label = label
        input = input.cuda()
        output = model(input).squeeze()
        correct += calculate_train_acc(output, label)
        count = i
    return float(correct) / float(count)

def train():
    train_x, test_x, train_y, test_y = data_util.load_data()
    train_data = DataSet(train_x, train_y)
    test_data = DataSet(test_x, test_y)
    epoch = 100
    batch_size = 1000
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MLPClassifier()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_acc_list = []
    test_acc_list = []
    for e in range(epoch):
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            output = model(inputs).squeeze()
            train_acc = calculate_train_acc(output, labels)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            test_acc = calculate_test_acc(model, test_data)
            print("train_acc: {}, test_acc: {}".format(train_acc, test_acc))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
    acc_lists_file = os.path.dirname(__file__) + "acc_lists.txt"
    with open(acc_lists_file, "rb") as f:
        pk.dump((train_acc_list, test_acc_list), f)
    torch.save(model.state_dict(), os.path.dirname(__file__) + "model.pt")

if __name__ == "__main__":
    train()