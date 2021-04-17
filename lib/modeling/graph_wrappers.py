import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .project_head import ProjectHead, ProjectionMLP, PredictionMLP


class ContrastWrapper(nn.Module):

    def __init__(self, encoder, hid_dim=128, head_type='mlp'):
        super(ContrastWrapper, self).__init__()

        self.encoder = encoder
        #dim_in = list(encoder.children())[-1].in_features
        in_dim = encoder.feature_dim

        self.proj_head = ProjectHead(in_dim, hid_dim, head_type)

    def forward(self, x, bb_grad=True):
        
        feat = self.encoder(x)
        feat = self.proj_head(feat)
        if not bb_grad:
            feat = feat.detach()

        return feat


# https://github.com/taoyang1122/pytorch-SimSiam/blob/main/models/simsiam.py
class SimSiam(nn.Module):

    def __init__(self, encoder, hid_dim=1024):
        super(SimSiam, self).__init__()

        self.encoder = encoder
        self.feature_dim = encoder.feature_dim

        # projection MLP
        self.projection = ProjectionMLP(self.feature_dim, hid_dim, hid_dim)
        # prediction MLP
        self.prediction = PredictionMLP(hid_dim, hid_dim//2, hid_dim)

        # negative cosine discriminative function
        self._parse_d()

        # self.reset_parameters()

    def forward(self, x):
        # x = self.encoder(x)
        # x = x.view(x.size(0), -1)
        # projection
        # hid_feat = self.projection(x)
        # prediction
        # logit = self.prediction(hid_feat)
        # return hid_feat, logit
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        x1 = self.encoder(x1)
        # projection
        hid_feat1 = self.projection(x1)
        # prediction
        logit1 = self.prediction(hid_feat1)

        x2 = self.encoder(x2)
        # projection
        hid_feat2 = self.projection(x2)
        # prediction
        logit2 = self.prediction(hid_feat2)

        loss = self.d(logit1, hid_feat2) / 2 + self.d(logit2, hid_feat1) / 2
        return loss

    def reset_parameters(self):
        # reset conv initialization to default uniform initialization
        from functools import reduce
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                n = reduce(lambda x, y: x*y, m.kernel_size) * m.in_channels
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def _parse_d(self):
        self.d = D()

class D(nn.Module):

    def __init__(self, fun_type='v2'):
        super(D, self).__init__()
        self.fun_type = fun_type

    def forward(self, logit, feat):
        if self.fun_type == 'v1':
            feat = feat.detach()
            logit = F.normalize(logit, dim=1)
            feat = F.normalize(feat, dim=1)
            return -(logit * feat).sum(dim=1).mean()
        elif self.fun_type == 'v2':
            return -F.cosine_similarity(logit, feat.detach(), dim=-1).mean()
        else: 
            raise ValueError('Unknown type in simsiam D!')        

class GraphWrapper(nn.Module):

    def __init__(self, encoder, hid_dim=1024, head_type='mlp', mem_type='simsiam'):
        super(GraphWrapper, self).__init__()
        if mem_type == 'simsiam':
            self.model = SimSiam(encoder=encoder, hid_dim=hid_dim)
        else:
            self.model = ContrastWrapper(encoder=encoder, hid_dim=hid_dim, head_type=head_type)

    def forward(self, x):
        return self.model(x)
