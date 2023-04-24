import numpy as np
import sklearn
import imblearn
import torch
import copy
import pickle as pk
import random
import os
from data_set import DataSet
from mlp import MLPClassifier
import train_mlp
import data_util

device = torch.device("cuda:0")
# device = torch.device("cpu")
print(device)

proj_dir = os.path.dirname(__file__)

"""
reference: 
https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2
https://github.com/udacity/deep-learning-v2-pytorch/blob/master/weight-initialization/weight_initialization_exercise.ipynb

"""
def calculate_test_acc(model, testing_data):
    model.to(device)
    return train_mlp.calculate_test_acc(model, testing_data)

def calculate_diff(model, testing_data):
    model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    batch_size = 1000
    testloader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True, num_workers=2)
    loss = 0
    count = 0
    for i, data in enumerate(testloader):
        input, label = data
        label = label.cuda()
        input = input.cuda()
        output = model(input).squeeze()
        loss += loss_fn(output, label).item()
        loss /= 2
    return -1 * loss

def confusion_matrix(model, test_data):
    model.to(device)
    batch_size = 1000
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
    actual = np.array([])
    predict = np.array([])
    for i, data in enumerate(testloader):
        input, label = data
        label = label.unsqueeze(-1)
        input = input.to(device)
        output = model(input)
        output = output.cpu().detach().numpy()
        output = np.round(output).astype(int).squeeze()
        label = label.cpu().detach().numpy().astype(int).squeeze()
        actual = np.concatenate((actual, label))
        predict = np.concatenate((predict, output))
    conf_mat = sklearn.metrics.confusion_matrix(actual, predict)
    return conf_mat

def training(exp_num, training_data, testing_data):
    # load all init weights
    population = []
    pop_fit_train = []
    pop_fit_test = []
    for i in range(20):
        model = MLPClassifier()
        fit_train = calculate_diff(model, training_data)
        fit_test = calculate_test_acc(model, testing_data)
        population.append(model)
        pop_fit_train.append(fit_train)
        pop_fit_test.append(fit_test)
    mutation_rate = 0.7
    cross_over_rate = 0.7
    print("init pop finished")
    diff_ea(exp_num, population, pop_fit_train, pop_fit_test, mutation_rate, cross_over_rate, training_data, testing_data)

def diff_ea(exp_num, init_pop, init_pop_fit_train, init_pop_fit_test, mutation_rate, cross_over_rate, training_data, testing_data):
    generation  = 100
    population = init_pop
    pop_fit_train = np.array(init_pop_fit_train)
    pop_fit_test = np.array(init_pop_fit_test)
    train_avg_acc = []
    train_max_acc = []
    test_avg_acc = []
    test_max_acc = []
    for g in range(generation):
        pop_indices = range(len(population))
        for i in pop_indices:
            prim_p = population[i]
            candidates = random.sample([item for item in range(len(population)) if not item == i], k=3)
            sec_p = mutation([population[idx] for idx in candidates], mutation_rate)
            child = cross_over(prim_p, sec_p, cross_over_rate)
            child_train_acc = calculate_test_acc(child, training_data)
            child_test_acc = calculate_test_acc(child, testing_data)
            population.append(child)
            pop_fit_train = np.append(pop_fit_train, child_train_acc)
            pop_fit_test = np.append(pop_fit_test, child_test_acc)
        survive_idx = np.argsort(pop_fit_train)[::-1][:int(len(population)/2)]
        survive = []
        for idx in survive_idx:
            survive.append(population[idx])
        population = survive
        pop_fit_train = pop_fit_train[survive_idx]
        pop_fit_test = pop_fit_test[survive_idx]
        train_max_acc.append(pop_fit_train[0])
        test_max_acc.append(pop_fit_test[0])
        train_avg = np.sum(pop_fit_train) / float(len(pop_fit_train))
        train_avg_acc.append(train_avg)
        test_avg = np.sum(pop_fit_test) / float(len(pop_fit_test))
        test_avg_acc.append(test_avg)
        print("train acc: {}, test acc: {}".format(train_avg, test_avg))
    save_accuracy_list(train_max_acc, train_avg_acc, test_max_acc, test_avg_acc, exp_num)
    pop_dir = proj_dir + "/result/model/exp_{}/".format(exp_num)
    if not os.path.exists(pop_dir):
        os.mkdir(pop_dir)
    with open(pop_dir + "models.mod", "wb") as f3:
        pk.dump(population, f3)

def mutation(candidates, mutation_rate):
    result_state_dict = {}
    for k in candidates[0].state_dict().keys():
        result_state_dict[k] = candidates[0].state_dict()[k] + mutation_rate * (candidates[1].state_dict()[k] + candidates[2].state_dict()[k])
    model = MLPClassifier()
    model.load_state_dict(result_state_dict)
    return model

def cross_over(prim_p, sec_p, cross_over_rate):
    child_state_dict = {}
    for k in prim_p.state_dict().keys():
        child_weight_k = torch.empty((prim_p.state_dict()[k].size())).flatten()
        prim_k_flatten = prim_p.state_dict()[k].flatten()
        sec_k_flatten = sec_p.state_dict()[k].flatten()
        for i in range(len(prim_k_flatten)):
            prob = random.random()
            child_weight_k[i] = prim_k_flatten[i] if prob < cross_over_rate else sec_k_flatten[i]
        child_weight_k = child_weight_k.reshape(prim_p.state_dict()[k].size())
        child_state_dict[k] = child_weight_k
    model = MLPClassifier()
    model.load_state_dict(child_state_dict)
    return model

def save_accuracy_list(train_max_acc, train_avg_acc, test_max_acc, test_avg_acc, exp_num):
    acc_dir = proj_dir + "/accuracy_list/exp_{}".format(exp_num)
    if not os.path.exists(acc_dir):
        os.mkdir(acc_dir)
    with open(acc_dir + "/train_max.txt", "wb") as f1:
        pk.dump(train_max_acc, f1)
    with open(acc_dir + "/train_avg.txt", "wb") as f2:
        pk.dump(train_avg_acc, f2)
    with open(acc_dir + "/test_max.txt", "wb") as f3:
        pk.dump(test_max_acc, f3)
    with open(acc_dir + "/test_avg.txt", "wb") as f4:
        pk.dump(test_avg_acc, f4)


def get_confuse_mat():
    x_train, x_test, y_train, y_test = data_util.load_data()
    train_data = DataSet(x_train, y_train)
    test_data = DataSet(x_test, y_test)
    experience_num = 10
    avg_confuse = None
    for i in range(experience_num):
        # training(i, train_data, test_data)
        exp_dir = proj_dir + "/result/model/exp_{}".format(i)
        confuse = None
        with open (exp_dir + "/models.mod", "rb") as f:
            models = pk.load(f)
        for m in models:
            if confuse is None:
                confuse = confusion_matrix(m, test_data)
            else:
                confuse += confusion_matrix(m, test_data)
        if avg_confuse is None:
            avg_confuse = confuse / float(len(models))
        else:
           avg_confuse += confuse / float(len(models)) 
    avg_confuse = avg_confuse / float(10)
    print(avg_confuse)

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = data_util.load_data()
    train_data = DataSet(x_train, y_train)
    test_data = DataSet(x_test, y_test)
    experience_num = 1
    for i in range(experience_num):
        print("#########exp {}#########".format(i))
        training(exp_num=i, training_data=train_data, testing_data=test_data)
    # get_confuse_mat()
