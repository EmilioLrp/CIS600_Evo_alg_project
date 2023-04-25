import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os

proj_dir = os.path.dirname(__file__)

def read_accuracies():
    sum_train_acc = None
    sum_test_acc = None
    for exp_num in range(10):
        with open(proj_dir + "/accuracies/mlp/exp_{}/acc_lists.txt".format(exp_num), "rb") as f:
            train_acc, test_acc = pk.load(f)
            if sum_train_acc is None:
                sum_train_acc = np.array(train_acc)
            else:
                sum_train_acc += np.array(train_acc)
            if sum_test_acc is None:
                sum_test_acc = np.array(test_acc)
            else:
                sum_test_acc += np.array(test_acc)
    avg_train_acc = sum_train_acc / float(10)
    avg_test_acc = sum_test_acc / float(10)
    x = np.log(np.arange(1, sum_train_acc.size + 1))
    plt.plot(x, avg_train_acc, label="train accuracy")
    plt.plot(x, avg_test_acc, label="test accuracy")
    plt.xlabel("iteration (log)")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    # for exp_num in range(10):
    #     x = None
    #     with open(proj_dir + "/accuracies/mlp/exp_{}/acc_lists.txt".format(exp_num), "rb") as f:
    #         train_acc, test_acc = pk.load(f)
    #     if x is None:
    #        x = range(len(train_acc))
    #     plt.plot(x, train_acc, label="exp_{}".format(exp_num))
    # plt.legend()
    # plt.show()

def read_confusion_matrix():
    mat = None
    for exp_num in range(1, 10):
        file_name = "confusion_matrix/exp_{}.npy".format(exp_num)
        if mat is None:
            mat = np.fromfile(file_name)
        else:
            mat += np.fromfile(file_name)
    print(mat / float(10))


if __name__ == '__main__':
    read_accuracies()
    # read_confusion_matrix()