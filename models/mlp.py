import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=4, activation=F.relu, output_dim=3, sigmoid_output=False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.depth = depth
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        
        return x
