import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Gate scores initialized to high values so sigmoid(gate) ~ 1 at start
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features) * 2.0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

    def forward(self, x):
        gates = self.get_gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

import math # Needed for reset_parameters

class PrunableMLP(nn.Module):
    def __init__(self, input_dim=3072, hidden1=512, hidden2=256, output_dim=10):
        super(PrunableMLP, self).__init__()
        self.layer1 = PrunableLinear(input_dim, hidden1)
        self.layer2 = PrunableLinear(hidden1, hidden2)
        self.layer3 = PrunableLinear(hidden2, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def get_all_gates(self):
        return [
            self.layer1.get_gates(),
            self.layer2.get_gates(),
            self.layer3.get_gates()
        ]
