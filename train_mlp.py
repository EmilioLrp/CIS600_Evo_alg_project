from data_set import DataSet
from mlp import MLPClassifier
import torch
import data_util
import numpy as np
import os 
import pickle as pk

def calculate_train_acc(pred, label):
    pred = pred.cpu().detach()
    pred = torch.nn.functional.softmax(pred, dim=1).numpy()
    max_idx = np.argmax(pred, -1)
    pred_round = np.zeros(pred.shape)
    for i in range(pred.shape[0]):
        pred_round[i, max_idx[i]] = float(1)
    label = label.cpu().detach().numpy()
    acc = 0
    for i in range(label.shape[0]):
        if np.array_equal(pred_round[i, :], label[i, :]):
            acc += 1
    return float(acc)/ float(label.shape[0])

def calculate_test_acc(model, testing_data):
    batch_size = 1000
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

def train(exp_num):
    accuracy_path = os.path.dirname(__file__) + "/accuracies/mlp/exp_{}/".format(exp_num)
    model_path = os.path.dirname(__file__) + "/models/mlp/exp_{}/".format(exp_num)
    if not os.path.exists(accuracy_path):
        os.makedirs(accuracy_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
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
    acc_lists_file = accuracy_path + "acc_lists.txt"
    with open(acc_lists_file, "wb") as f:
        pk.dump((train_acc_list, test_acc_list), f)
    torch.save(model.state_dict(), model_path + "model.pt")

if __name__ == "__main__":
    exp_num = 10
    for exp in range(exp_num):
        train(exp)