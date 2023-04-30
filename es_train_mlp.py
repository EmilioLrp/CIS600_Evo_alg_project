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
import matplotlib.pyplot as plt


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

def training(exp_num, training_data, testing_data):
    print("********experiment {}*********".format(exp_num))
    pop_size = 15
    offspring_size = 100
    population = []
    # init_nums = np.random.choice(range(10), pop_size, replace=False)
    for i in range(pop_size):
        model = MLPClassifier()
        population.append(model)
    result = evolution_alg(pop_size, offspring_size, population, training_data, testing_data, exp_num)
    exp_dir = proj_dir + "/model/es_1/exp_{}".format(exp_num)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    for i in range(len(result)):
        torch.save(result[i].state_dict(), exp_dir + "/model_{}.pt".format(i))


def evolution_alg(pop_size, offspring_size, init_pop, evaluate_data, test_data, exp_num):
    """
    reference https://github.com/shahril96/neural-network-with-genetic-algorithm-optimizer
    """
    population = init_pop
    sigma = float(100)
    n = 3
    k_lim = 50
    # k_lim = 1
    # crossover rate
    p = 0.3
    # ration for self adaptive sigma
    r = 0.1
    pop_train_acc = [acc_estimate(item, evaluate_data) for item in population]
    # pop_test_acc = [acc_estimate(item, test_data) for item in population]
    # train_max_acc_list = []
    train_avg_acc_list = []
    # test_max_acc_list = []
    test_avg_acc_list = []
    mutation_rate = 0.5
    for k in range(k_lim):
        counter = 0
        for i in range(n):
            offsprings = []
            o_performance = np.array([])
            for _ in range(offspring_size):
                # sample parents without replacement
                parents = random.sample(range(len(population)), 2)
                p1 = population[parents[0]]
                p2 = population[parents[1]]
                ref_performance = pop_train_acc[parents[0]]
                child = one_pt_crossover(p, p1, p2)
                child = mutation(mutation_rate, sigma, child)
                train_acc = acc_estimate(child, evaluate_data)
                if train_acc > ref_performance:
                    counter += 1
                offsprings.append(child)
                o_performance = np.append(o_performance, train_acc)
            offsprings += population
            o_performance = np.append(o_performance, np.array(pop_train_acc))
            best_o = np.flip(np.argsort(o_performance))[:pop_size]
            train_avg_acc = np.sum(o_performance[best_o]) / float(pop_size)
            train_avg_acc_list.append(train_avg_acc)
            population = []
            pop_train_acc = []
            test_avg_acc = 0
            for idx in best_o:
                net = offsprings[idx]
                test_acc = calculate_test_acc(net, test_data)
                test_avg_acc += test_acc
                population.append(net)
                pop_train_acc.append(o_performance[idx])
            test_avg_acc = test_avg_acc / float(pop_size)
            test_avg_acc_list.append(test_avg_acc)
            print("iter: {}, train_acc: {}, test_acc: {}, best_acc:, {}, sigma: {}".format(i+k*n, train_avg_acc, test_avg_acc, o_performance[best_o[0]], sigma))
        if counter < n * offspring_size / 20:
            sigma = sigma * (1-r)
        else:
            sigma = sigma * (1+r)
    save_accuracy_list(None, train_avg_acc_list, None, test_avg_acc_list, exp_num)    
    return population

def save_accuracy_list(train_max_acc, train_avg_acc, test_max_acc, test_avg_acc, exp_num):
    acc_dir = proj_dir + "/accuracies/es_1/exp_{}".format(exp_num)
    if not os.path.exists(acc_dir):
        os.makedirs(acc_dir)
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
    p1_states = copy.deepcopy(parent_1.state_dict())
    p2_states = copy.deepcopy(parent_2.state_dict())

    # for key in p1_states.keys():
    #     p1_weights, p2_weights = p1_states[key], p2_states[key]
    #     dim = p1_weights.size()
    #     p1_w_flat, p2_w_flat = torch.flatten(p1_weights), torch.flatten(p2_weights)
    #     x_point = np.random.randint(low = 0, high = len(p1_w_flat), size=1)[0]
    #     r = random.random()
    #     if r > cross_over_rate:
    #         p1_w_flat = torch.cat((p1_w_flat[:x_point], p2_w_flat[x_point:]))
    #         p1_states[key] = p1_w_flat.reshape(dim)
    r = random.random()
    child_state = {}
    if r < cross_over_rate:
        child_state = random.choice([p1_states, p2_states])
    else: 
        for k in p1_states.keys():
            child_state[k] = (p1_states[k] + p2_states[k]) / 2
    model = MLPClassifier()
    model.load_state_dict(child_state)
    return model


def mutation(mutation_rate: float, sigma: float, offspring: MLPClassifier):
    states = copy.deepcopy(offspring.state_dict())
    distribution = torch.distributions.Cauchy(0, sigma)
    for key in states.keys():
        weights = states[key]
        dim = weights.size()
        # weight_flatten, mut_rate_flatten = torch.flatten(weights), torch.flatten(mutation_rate[key])
        weight_flatten = torch.flatten(weights)
        for i in range(len(weight_flatten)):
            r = random.random()
            if r > mutation_rate:
                # weight_flatten[i] += torch.normal(0, sigma, size=weight_flatten[i].size())
                weight_flatten[i] += distribution.sample().item()
        states[key] = weight_flatten.reshape(dim)
    model = MLPClassifier()
    model.load_state_dict(states)
    return model

def confusion_matrix(model, test_data):
    model.cuda()
    batch_size = 1000
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
    actual = None
    predict = None
    for i, data in enumerate(testloader):
        input, label = data
        input = input.cuda()
        output = model(input)
        output = output.cpu().detach()
        # output = np.round(output).astype(int).squeeze()
        pred = torch.nn.functional.softmax(output, dim=1).numpy()
        max_idx = np.argmax(pred, -1)
        # pred_round = np.zeros(pred.shape)
        # for i in range(pred.shape[0]):
        #     pred_round[i, max_idx[i]] = float(1)
        label = label.cpu().detach().numpy().astype(int)
        label_max = np.argmax(label, -1)
        if actual is None:
            actual = label_max
        else:
            actual = np.concatenate((actual, label_max))
        if predict is None:
            predict = max_idx
        else:
            predict = np.concatenate((predict, max_idx))
    conf_mat = sklearn.metrics.confusion_matrix(actual, predict, labels=range(26))
    return conf_mat


def get_confuse_mat():
    # accuracy_path = os.path.dirname(__file__) + "/accuracies/mlp/exp_{}/"
    train_x, test_x, train_y, test_y = load_data()
    train_data = DataSet(train_x, train_y)
    test_data = DataSet(test_x, test_y)
    experience_num = 10
    avg_confuse = None
    for i in range(experience_num):
        exp_dir = proj_dir + "/model/es/exp_{}".format(i)
        confuse = None
        # with open (model_path + "/models.mod", "rb") as f:
        for j in range(25):
            model_path = exp_dir + "/model_{}.pt".format(j)
            model = MLPClassifier()
            model.load_state_dict(torch.load(model_path))
            
            if confuse is None:
                confuse = confusion_matrix(model, test_data)
            else:
                confuse += confusion_matrix(model, test_data)
        confuse = confuse / float(25)
        if avg_confuse is None:
            avg_confuse = confuse
        else:
            avg_confuse  = avg_confuse + confuse
    avg_confuse = avg_confuse / float(10)
    disp = sklearn.metrics.ConfusionMatrixDisplay(confuse, display_labels=range(26))
    disp.plot()
    plt.show()
    # print(avg_confuse)

if __name__ == '__main__':
    # x_train, x_test, y_train, y_test = load_data()
    # train_data = DataSet(x_train, y_train)
    # test_data = DataSet(x_test, y_test)
    # experience_num = 10
    # avg_confuse = None
    # for i in range(experience_num):
    #     training(i, train_data, test_data)
    get_confuse_mat()
