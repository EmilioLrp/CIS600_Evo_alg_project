import torch

class MLPClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(16, 32)
        self.l2 = torch.nn.Linear(32, 64)
        self.l5 = torch.nn.Linear(64, 32)
        self.l6 = torch.nn.Linear(32, 26)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.relu(self.l5(x))
        x = self.l6(x)
        return x