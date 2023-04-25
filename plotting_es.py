import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os

proj_dir = os.path.dirname(__file__)


def read_accuracies():
    # sum_train_max_acc = None
    # sum_test_max_acc = None
    sum_train_avg_acc = None
    sum_test_avg_acc = None
    for exp_num in range(10):
        acc_dir = proj_dir + "/accuracies/es/exp_{}".format(exp_num)
        # with open(acc_dir + "/train_max.txt", "rb") as f1:
        #     train_max_acc = pk.load(f1)
        #     if sum_train_max_acc is None:
        #         sum_train_max_acc = np.array(train_max_acc)
        #     else:
        #         sum_train_max_acc += np.array(train_max_acc)
        with open(acc_dir + "/train_avg.txt", "rb") as f2:
            train_avg_acc = pk.load(f2)
            if sum_train_avg_acc is None:
                sum_train_avg_acc = np.array(train_avg_acc)
            else:
                sum_train_avg_acc += np.array(train_avg_acc)
        # with open(acc_dir + "/test_max.txt", "rb") as f3:
        #     test_max_acc = pk.load(f3)
        #     if sum_test_max_acc is None:
        #         sum_test_max_acc = np.array(test_max_acc)
        #     else:
        #         sum_test_max_acc += np.array(test_max_acc)
        with open(acc_dir + "/test_avg.txt", "rb") as f4:
            test_avg_acc = pk.load(f4)
            if sum_test_avg_acc is None:
                sum_test_avg_acc = np.array(test_avg_acc)
            else:
                sum_test_avg_acc += np.array(test_avg_acc)
    # avg_train_max_acc = sum_train_max_acc / float(10)
    # avg_test_max_acc = sum_test_max_acc / float(10)
    avg_train_avg_acc = sum_train_avg_acc / float(10)
    avg_test_avg_acc = sum_test_avg_acc / float(10)
    x = np.arange(1, sum_train_avg_acc.size + 1)
    # plt.plot(x, avg_train_max_acc, label="max train accuracy")
    # plt.plot(x, avg_test_max_acc, label="max test accuracy")
    plt.plot(x, avg_train_avg_acc, label="avg train accuracy")
    plt.plot(x, avg_test_avg_acc, label="avg test accuracy")
    plt.xlabel("generation")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    # plt.savefig(proj_dir + "accuracies.png")


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