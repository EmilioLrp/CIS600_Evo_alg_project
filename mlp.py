import torch

class MLPClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(16, 32)
        self.l2 = torch.nn.Linear(32, 64)
        self.l3 = torch.nn.Linear(64, 32)
        self.l4 = torch.nn.Linear(32, 26)
    
    def foward(self, x):
        x = self.l1(x)
        x = torch.nn.ReLU()(x)
        x = self.l2(x)
        x = torch.nn.ReLU(x)
        x = self.l3(x)
        x = torch.nn.ReLU(x)
        x = self.l4(x)
        return torch.nn.Softmax(x)