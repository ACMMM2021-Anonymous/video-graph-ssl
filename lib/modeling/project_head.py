import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)

class ProjectHead(nn.Module):

    def __init__(self, in_dim, feat_dim=128, head_type='mlp'):
        super(ProjectHead, self).__init__()

        if head_type == 'linear':
            self.head = nn.Sequential(
                nn.Linear(in_dim, feat_dim),
                Normalize(2)
            )
        elif head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head_type))

    def forward(self, x):
        return self.head(x)

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x