import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from entmax import sparsemax, entmax15

class DynamicSparseGate(nn.Module):
    def __init__(self, in_channels):
        super(DynamicSparseGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout=0.0, factor_scaling=True, bias=True):
        super(LinearEncoder, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.rank = feature_dim
        self.factor_scaling = factor_scaling

        self.U = nn.Parameter(torch.empty(feature_dim, self.rank))
        self.V = nn.Parameter(torch.empty(feature_dim, self.rank))

        self.dropout_layer = nn.Dropout(dropout)
        self.lin_class = weight_norm(nn.Linear(input_dim, feature_dim))

        self.sparsity_gate = DynamicSparseGate(input_dim)

        self.norm = nn.LayerNorm(feature_dim)
        self.eps = 1e-6

        if bias:
            self.bias = nn.Parameter(torch.zeros(feature_dim))

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_class.weight)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, x):
        # linear projection
        z = self.lin_class(x)

        # 1. Bilinear attention factorization
        z_u = torch.mm(z, self.U)  # [N, rank]
        z_v = torch.mm(z, self.V)  # [N, rank]

        # self-attention matrix: e = z_u @ z_v.T
        sim = torch.mm(z_u, z_v.t()) / (self.rank ** 0.5)  # [N, N]
        # sim = torch.mm(z_u, z_v.t())

        # 2. Dynamic sparsity gating: calculating the dynamic sparsity factor
        tau = self.sparsity_gate(x)  # [N, 1]

        # sim = sim * (1.0 - tau)
        if self.factor_scaling:
            sim = sim * (1.0 - tau)
        else:
            sim = sim / (1.0 - tau + self.eps) # performance remains competitive
        # sim = sim.fill_diagonal_(0.0)

        # 3. Structured sparse projection
        attention = entmax15(sim, dim=1)

        # dropout
        attention = self.dropout_layer(attention)

        # residual attention updated
        z = z + attention @ z

        if hasattr(self, 'bias'):
            z = z + self.bias

        # classification probabilities
        labels = F.softmax(z, dim=-1)

        return labels, attention
