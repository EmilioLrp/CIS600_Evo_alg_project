import os 
import numpy as np
import sklearn.model_selection
import imblearn

data_dir = os.path.dirname(__file__) + "/letter_data" 

def load_data():
    data = []
    labels = []
    with open(data_dir + "/letter-recognition.data", "r") as f:
        data_lines = f.readlines()
        for l in data_lines:
            l = l.split("\n")[0]
            l_split = l.split(",")
            label = np.zeros(26)
            idx = ord(l_split[0]) - ord('A')
            label[idx] = 1
            labels.append(label)
            inputs = np.array([float(item) for item in l_split[1:]])
            data.append(inputs)
            # data.append((idx, inputs))
    return sklearn.model_selection.train_test_split(data, labels, test_size=0.3, random_state=42)

def load_data_mat():
    train_x, test_x, train_y, test_y = load_data()
    return np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)

def load_data_by_class():
    data_dict = load_data_dict()
    dataset_dict = {}
    for i in range(26):
        true_set = data_dict[i]
        false_set = []
        for j in range(26):
            if not i == j:
                false_set += data_dict[j]
        true_labels = [1] * len(true_set)
        false_labels = [0] * len(false_set)
        rus = imblearn.under_sampling.RandomUnderSampler(random_state=42)
        features_rus, labels_rus = rus.fit_resample(true_set + false_set, true_labels + false_labels)
        dataset_dict[i] = sklearn.model_selection.train_test_split(features_rus, labels_rus, test_size=0.3, random_state=42)
    return dataset_dict

def load_data_dict():
    data_dict = {}
    with open(data_dir + "/letter-recognition.data", "r") as f:
        data_lines = f.readlines()
        for l in data_lines:
            l = l.split("\n")[0]
            l_split = l.split(",")
            # label = np.zeros(26)
            idx = ord(l_split[0]) - ord('A')
            # label[idx] = 1
            inputs = np.array([float(item) for item in l_split[1:]])
            if idx not in data_dict:
                data_dict[idx] = []
            data_dict[idx].append(inputs)
            # data.append(inputs)
    return data_dict


if __name__ == "__main__":
    data_dict = load_data_by_class()
    print("sss")