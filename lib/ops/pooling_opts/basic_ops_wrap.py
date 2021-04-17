import torch
import torch.nn as nn

class TemporalAggreModel(nn.Module):

    def __init__(self, pooling='avg', model_type='2D'):
        super(TemporalAggreModel, self).__init__()
        self.model_type = model_type
        self.pooling = pooling
        if pooling == 'avg':
            self.agg = torch.mean
        elif pooling == 'max':
            self.agg = torch.max
        elif pooling == 'lstm':
            self.agg = nn.LSTM(m, n, h)
        
        if model_type == '2D':
            self.dim = 1
        elif model_type == '3D':
            self.dim=2

    def forward(self, x):
        if self.pooling == 'avg' or self.pooling == 'max':
            y = self.agg(x, dim=self.dim)
        else:
            y = self.agg(x)
        return y
