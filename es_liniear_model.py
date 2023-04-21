import data_util
import numpy as np
import random
import torch

def mutation(child, sigma, mut_rate):
    shape = child.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            r = random.random()
            if r > mut_rate[i, j]:
                child[i, j] += np.random.normal(0, sigma)
    return child

def crossover(p1, p2):
    shape = p1.shape
    p1_flatten = p1.flatten()
    p2_flatten = p2.flatten()
    # idx = random.randint(0, len(p1_flatten) - 1)
    # child = np.concatenate((p1_flatten[:idx], p2_flatten[idx:]))
    child = p1_flatten + p2_flatten
    child = np.reshape(child, shape)
    return child

def calculate_accuracy(individual, input, label):
    prediction = torch.Tensor(np.matmul(input, individual))
    prediction = torch.nn.functional.softmax(prediction).numpy()
    max_idx = np.argmax(prediction, -1)
    pred_round = np.zeros(prediction.shape)
    for i in range(prediction.shape[0]):
        pred_round[i, max_idx[i]] = float(1)
    acc = 0
    for i in range(len(label)):
        if np.array_equal(pred_round[i, :], label[i]):
            acc += 1
    return float(acc)/ float(len(label))

def calculate_loss(individual, input, label):
    prediction = torch.Tensor(np.matmul(input, individual))
    prediction = torch.nn.functional.softmax(prediction).numpy()
    label_np = np.array(label)
    loss_fn = torch.nn.MSELoss()
    return  - loss_fn(torch.Tensor(prediction), torch.Tensor(label_np)).item()

def es(init_pop, pop_size, ind_shape, train_x, test_x, train_y, test_y):
    # generations = 100
    k_lim = 20
    n = 5
    r = 0.1
    offsprings = 40
    population = init_pop
    pop_acc = [calculate_loss(item, train_x, train_y) for item in population]
    sigma = float(5)
    mut_rate = (np.random.rand(ind_shape[0], ind_shape[1]))
    cross_rate = 0.7
    for k in range(k_lim):
        counter = 0
        for i in range(n):
            children = []
            children_acc = []
            for o in range(offsprings):
                choices = random.sample(population, 2)
                p1 = choices[0]
                p2 = choices[1]
                x = random.random()
                if x > cross_rate:
                    child = crossover(p1, p2)
                else:
                    child = p1
                child = mutation(child, sigma, mut_rate)
                children.append(child)
                child_acc = calculate_loss(child, train_x, train_y)
                children_acc.append(child_acc)
                if child_acc > calculate_loss(p1, train_x, train_y):
                    counter += 1
            # mu, lambda ES:
            best_o = np.flip(np.argsort(children_acc))[:pop_size]
            population = []
            pop_acc = []
            for o in best_o:
                population.append(children[o])
                pop_acc.append(children_acc[o])
            print("acc: {}".format(float(sum(pop_acc)) / pop_size))
        if counter < n * offsprings / 5:
            sigma = sigma * (1-r)
        else:
            sigma = sigma * (1+r)
    return population

def onehot_transform(label, idx):
    onehot_label = np.zeros(26)
    onehot_label[idx] = 1
    if label == 1:
        return onehot_label
    else:
        return np.zeros(26)

def train():
    data_dict = data_util.load_data_by_class()
    models = []
    indices = sorted(list(data_dict.keys()))
    for idx in indices:
        print("training idx {}".format(idx))
        train_x, test_x, train_y, test_y = data_dict[idx]
        train_y_onehot = [onehot_transform(item, idx) for item in train_y]
        test_y_onehot = [onehot_transform(item, idx) for item in test_y]
        population = []
        pop_size = 5
        ind_shape = (16, 26)
        for i in range(pop_size):
            p = np.random.uniform(-10, 10, ind_shape)
            population.append(p)
        population = es(population, pop_size, ind_shape, train_x, test_x, train_y_onehot, test_y_onehot)
        models.append(population)

if __name__=="__main__":
    train()