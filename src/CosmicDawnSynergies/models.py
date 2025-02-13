import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim = 11, hidden_dim = 100, n_hidden = 4, out_dim = 1, **kwargs):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        x = x.squeeze(-1)
        return x
