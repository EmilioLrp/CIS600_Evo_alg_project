import os 
import numpy as np
import sklearn.model_selection
import torch

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
            inputs = np.array(l_split[1:])
            data.append(inputs)
            # data.append((idx, inputs))
    return sklearn.model_selection.train_test_split(data, labels, test_size=0.3, random_state=42)

def claculate_loss(model, testing_data, batch_size):
    loss_fn = torch.nn.CrossEntropyLoss(reduce="mean")

if __name__ == "__main__":
    train_x, test_x, train_y, test_y = load_data()
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y)