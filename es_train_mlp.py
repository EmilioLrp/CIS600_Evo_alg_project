import numpy as np
import sklearn
import torch
import copy
import pickle as pk
import random
import os
from data_set import DataSet
from mlp import MLPClassifier
from train_mlp import calculate_test_acc
from data_util import load_data

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

def acc_estimate(model, dataset):
    model.cuda()
    return calculate_test_acc(model, dataset)

def confusion_matrix(exp_num, model, test_data):
    model.to(device)
    batch_size = 100
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
    # np.save("confusion_matrix/exp_{}".format(exp_num), conf_mat)
    return conf_mat


def training(exp_num, training_data, testing_data):
    print("********experiment {}*********".format(exp_num))
    pop_size = 4
    offspring_size = 20
    population = []
    init_nums = np.random.choice(range(10), pop_size, replace=False)
    for i in init_nums:
        model = MLPClassifier()
        population.append(model)
    result = evolution_alg(pop_size, offspring_size, population, training_data, testing_data, exp_num)
    exp_dir = proj_dir + "/model/es/exp_{}".format(exp_num)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    for i in range(len(result)):
        torch.save(result[i].state_dict(), exp_dir + "/model_{}.pt".format(i))


def evolution_alg(pop_size, offspring_size, init_pop, evaluate_data, test_data, exp_num):
    """
    reference https://github.com/shahril96/neural-network-with-genetic-algorithm-optimizer
    """
    population = init_pop
    sigma = float(1)
    n = 10
    k_lim = 10
    p = 0.5
    r = 0.1
    # train_max_acc_list = []
    train_avg_acc_list = [acc_estimate(item, evaluate_data) for item in population]
    # test_max_acc_list = []
    test_avg_acc_list = [acc_estimate(item, test_data) for item in population]
    for k in range(k_lim):
        counter = 0
        for i in range(n):
            offsprings = []
            o_performance = np.array([])
            for _ in range(offspring_size):
                mutation_rate = {key:torch.rand(size=population[0].state_dict()[key].size()) for key in population[0].state_dict().keys()}
                # sample parents without replacement
                parents = random.sample(range(len(population)), 2)
                p1 = population[parents[0]]
                p2 = population[parents[1]]
                ref_performance = train_avg_acc_list[parents[0]]
                child = one_pt_crossover(p, p1, p2)
                child = mutation(mutation_rate, sigma, child)
                train_acc = acc_estimate(child, evaluate_data)
                if train_acc > ref_performance:
                    counter += 1
                offsprings.append(child)
                o_performance = np.append(o_performance, train_acc)
            best_o = np.flip(np.argsort(o_performance))[:pop_size]
            train_avg_acc = np.sum(o_performance[best_o]) / float(pop_size)
            train_avg_acc_list.append(train_avg_acc)
            population = []
            test_avg_acc = 0
            for idx in best_o:
                net = offsprings[idx]
                test_acc = calculate_test_acc(net, test_data)
                test_avg_acc += test_acc
                population.append(net)
            test_avg_acc = test_avg_acc / float(pop_size)
            test_avg_acc_list.append(test_avg_acc)
            print("iter: {}, train_acc: {}, test_acc: {}".format(i+k*n, train_avg_acc, test_avg_acc))
        if counter < n * offspring_size / 5:
            sigma = sigma * (1-r)
        else:
            sigma = sigma * (1+r)
    save_accuracy_list(None, train_avg_acc_list, None, test_avg_acc_list, exp_num)    
    return population

def save_accuracy_list(train_max_acc, train_avg_acc, test_max_acc, test_avg_acc, exp_num):
    acc_dir = proj_dir + "/accuracy_list/exp_{}".format(exp_num)
    if not os.path.exists(acc_dir):
        os.mkdir(acc_dir)
    if not train_max_acc is None:
        with open(acc_dir + "/train_max.txt", "wb") as f1:
            pk.dump(train_max_acc, f1)
    if not train_avg_acc is None:
        with open(acc_dir + "/train_avg.txt", "wb") as f2:
            pk.dump(train_avg_acc, f2)
    if not test_max_acc is None:
        with open(acc_dir + "/test_max.txt", "wb") as f3:
            pk.dump(test_max_acc, f3)
    if not test_avg_acc is None:
        with open(acc_dir + "/test_avg.txt", "wb") as f4:
            pk.dump(test_avg_acc, f4)

def one_pt_crossover(cross_over_rate: float, parent_1, parent_2):
    """
    linear.weight (24,12)
    linear.bias (24)
    hidden.weight (1, 24)
    hidden.bias (1)
    """
    p1_states = copy.deepcopy(parent_1.state_dict())
    p2_states = copy.deepcopy(parent_2.state_dict())

    for key in p1_states.keys():
        p1_weights, p2_weights = p1_states[key], p2_states[key]
        dim = p1_weights.size()
        p1_w_flat, p2_w_flat = torch.flatten(p1_weights), torch.flatten(p2_weights)
        x_point = np.random.randint(low = 0, high = len(p1_w_flat), size=1)[0]
        r = random.random()
        if r > cross_over_rate:
            p1_w_flat = torch.cat((p1_w_flat[:x_point], p2_w_flat[x_point:]))
            p1_states[key] = p1_w_flat.reshape(dim)
    model = MLPClassifier()
    model.load_state_dict(p1_states)
    return model


def mutation(mutation_rate: float, sigma: float, offspring: MLPClassifier):
    states = copy.deepcopy(offspring.state_dict())
    for key in states.keys():
        weights = states[key]
        dim = weights.size()
        weight_flatten, mut_rate_flatten = torch.flatten(weights), torch.flatten(mutation_rate[key])
        for i in range(len(weight_flatten)):
            r = random.random()
            if r > mut_rate_flatten[i]:
                weight_flatten[i] += torch.normal(0, sigma, size=weight_flatten[i].size())
        states[key] = weight_flatten.reshape(dim)
    model = MLPClassifier()
    model.load_state_dict(states)
    return model

def get_confuse_mat():
    x_train, y_train, x_test, y_test = load_data()
    train_data = DataSet(x_train, y_train)
    test_data = DataSet(x_test, y_test)
    experience_num = 10
    avg_confuse = None
    for i in range(experience_num):
        # training(i, train_data, test_data)
        exp_dir = proj_dir + "/result/model/exp_{}".format(i)
        confuse = None
        for j in range(4):
            model = MLPClassifier()
            model_file = exp_dir + "/model_{}.pt".format(j)
            model.load_state_dict(torch.load(model_file))
            if confuse is None:
                confuse = confusion_matrix(i, model, test_data)
            else:
                confuse += confusion_matrix(i, model, test_data)
        if avg_confuse is None:
            avg_confuse = confuse / float(4)
        else:
           avg_confuse += confuse / float(4) 
    avg_confuse = avg_confuse / float(10)
    print(avg_confuse)

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    train_data = DataSet(x_train, y_train)
    test_data = DataSet(x_test, y_test)
    experience_num = 10
    avg_confuse = None
    for i in range(experience_num):
        training(i, train_data, test_data)
    # get_confuse_mat()
