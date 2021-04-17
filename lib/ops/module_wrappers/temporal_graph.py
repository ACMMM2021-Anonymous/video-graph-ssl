import torch
from torch import nn
from torch.nn import functional as F
import math


class TemporalGraph(object):
    def __init__(self,
                 tem_len=16,
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = tem_len

        self.edge = self._parser_edge(tem_len)
        self.temporal_graph = self._parser_graph(
            self.num_node, self.edge, max_hop=max_hop)

    def _parser_edge(self, tem_len):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(i, i+1) for i in range(self.num_node-1)]
        return self_link + neighbor_link

    def _parser_graph(self, num_node, edge, max_hop=1):
        A = torch.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        hop_dis = torch.zeros((num_node, num_node)) + float('inf')
        transfer_mat = [torch.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (torch.stack(transfer_mat, dim=0) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

class GCN(nn.Module):

    def __init__(self, in_features, out_features=None, bias=False, skip=True):
        super(GCN, self).__init__()
        self.skip = skip
        self.in_features = in_features
        self.out_features = in_features if out_features is None else out_features
        
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=self.out_features,
                        kernel_size=(1, 1, 1), bias=bias)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.conv.weight.size(1))  
        self.conv.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.conv.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        
        support = self.conv(x)
        output = torch.einsum('bij,bcjhw->bcihw', (adj, support))
        
        if self.skip:
            output += support

        return output

class TemporalGraphAug(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, 
                bias=False, bn_layer=False, zero_init=False, max_pool=True,
                mask_frame=False, nei_size=None, alpah=0.5,
                num_gcn_layers=1, temperature=1., max_hop=3):
        super(TemporalGraphAug, self).__init__()

        self.sub_sample = sub_sample
        self.bias = bias
        self.bn_layer = bn_layer
        self.zero_init = zero_init
        self.max_pool = max_pool

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.mask_frame = mask_frame
        self.nei_size = nei_size
        self.alpha = alpah
        self.num_gcn_layers = num_gcn_layers
        self.temperature = temperature
        self.max_hop = max_hop

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.gcns = nn.ModuleList(
            [GCN(in_features=in_channels, out_features=inter_channels)]
        )
        for i in range(1, num_gcn_layers):
            gcn_layer = GCN(in_features=inter_channels, out_features=in_channels) if i == num_gcn_layers-1 else GCN(in_features=inter_channels)
            self.gcns.append(gcn_layer)
        
        
        pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2)) if self.max_pool else nn.AvgPool3d(kernel_size=(1, 2, 2))           
            

        if bn_layer:
            self.g_q = nn.Sequential(
                nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0, bias=self.bias),
                nn.BatchNorm3d(self.inter_channels)
            )
            self.g_k = nn.Sequential(
                nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0, bias=self.bias),
                nn.BatchNorm3d(self.inter_channels)
            )

            self.reset_parameter(self.g_q[0], self.g_k[0])
        else:
            self.g_q = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=self.bias)
            self.g_k = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=self.bias)

            self.reset_parameter(self.g_q, self.g_k)


        if sub_sample:
            self.g_q = nn.Sequential(self.g_q, pool_layer)
            self.g_k = nn.Sequential(self.g_k, pool_layer)

    def reset_parameters(self, m1, m2):
        if self.zero_init:
            nn.init.constant_(m1.weight, 0)
            nn.init.constant_(m2.weight, 0)
            if self.bias:
                nn.init.constant_(m1.bias, 0)
                nn.init.constant_(m2.bias, 0)
        else:
            n1 = m1.kernel_size[0] * m1.kernel_size[1] * m1.kernel_size[2] * m1.in_channels
            n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.in_channels
            stdv1 = 1. / math.sqrt(n1)
            stdv2 = 1. / math.sqrt(n2)
            m1.weight.data.uniform_(-stdv1, stdv1)
            m2.weight.data.uniform_(-stdv2, stdv2)
            if self.bias:
                m1.bias.data.uniform_(-stdv1, stdv1)
                m2.bias.data.uniform_(-stdv2, stdv2)
       
    
    def _get_sim_adj(self, x):
        '''
        :param x: (b, c, t, h, w)   3d case
        :return:
        '''

        batch_size = x.size(0)
        sequence_len = x.size(2)

        g_q = self.g_q(x)
        g_k = self.g_k(x)
        g_q = g_q.transpose(2, 1)
        g_k = g_k.transpose(2, 1)

        g_q = g_q.contiguous().view(batch_size, sequence_len, -1)
        g_k = g_k.contiguous().view(batch_size, sequence_len, -1)

        adjacent_matrix = torch.matmul(g_q, g_k.permute(0, 2, 1))

        if self.mask_frame:
            s_mask = torch.ones_like(adjacent_matrix).bool()  # [B, T1, T2]
            self.nei_size = sequence_len if not self.nei_size else self.nei_size
            for i, (l_1, l_2) in enumerate(zip(range(self.nei_size), range(self.nei_size))):
                s_mask[i][:l_1, :l_2] = 0
            adjacent_matrix.data.masked_fill_(s_mask.data, -float("inf"))

        adjacent_matrix = F.softmax(adjacent_matrix, dim=-1)

        return adjacent_matrix

    def _sample_adj_with_gussion(self, adj):
        if self.training:
            epsilon = torch.randn(adj.size()) * self.alpha + 1
            return adj * epsilon
        else:
            return adj

    def _sample_adj_with_rel_ber(self, adj):
        relax_ber = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temperature, probs=adj
        )
        resampled_adj = relax_ber.rsample()
        return resampled_adj

    def _sample_adj_with_van(self, adj):
        relax_ber = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temperature, probs=adj
        )
        sampled_adj = relax_ber.sample()
        return sampled_adj

    def _sample_adj(self, adj):
        pass

    def _parser_temporal_graph(self, sim_graph, hop_graph):
        
        theta = lambda x: math.exp(-x) / (1 + math.exp(-x) ** 2) + self.alpha  
        adj = torch.zeros_like(sim_graph)
        for hop in range(self.max_hop+1):
            adj[:, hop_graph == hop] = sim_graph[:, hop_graph == hop] * theta(hop)
        return adj

    def compute_ppr(self, adj, alpha=0.2):
        from scipy.linalg import fractional_matrix_power, inv

        d = torch.diag(adj.sum(dim=1))                                # D^ = Sigma A^_ii
        dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
        dinv = torch.from_numpy(dinv)
        at = dinv.mm(adj).mm(dinv)                                    # A~ = D^(-1/2) x A^ x D^(-1/2)
        return alpha * torch.from_numpy(inv((torch.eye(adj.shape[0]) - (1 - alpha) * at)))   # a(I_n-(1-a)A~)^-1

    def compute_heat(self, adj, t=5):
        from scipy.linalg import inv

        d = torch.diag(adj.sum(dim=1))
        return torch.exp(t * adj.mm(torch.from_numpy(inv(d)) - 1))        

    def forward(self, x):

        sequence_len = x.size(2)
        multi_hop_graph = TemporalGraph(tem_len=sequence_len, max_hop=self.max_hop)
        temporal_graph = multi_hop_graph.temporal_graph
        sim_graph = self._get_sim_adj(x)

        adj = self._parser_temporal_graph(sim_graph, temporal_graph)
        adj = self._sample_adj_with_rel_ber(adj)

        for gcn in self.gcns:
            x = gcn(x, adj)
        return x
