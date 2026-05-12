import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from layers import *

class SAGL(nn.Module):
    def __init__(self, input_dims, feature_dim, dropout, factor_scaling):
        super(SAGL, self).__init__()
        self.encoders = nn.ModuleList()
        for input_dim in input_dims:
            self.encoders.append(LinearEncoder(input_dim, feature_dim, dropout, factor_scaling))

    def forward(self, data_sets):
        label_set = []
        attention_set = []

        for encoder, data in zip(self.encoders, data_sets):
            labels, attention = encoder(data)
            label_set.append(labels)
            attention_set.append(attention)

        labels = torch.mean(torch.stack(label_set), dim=0)  # shape of (N, C)

        return labels, label_set, attention_set
