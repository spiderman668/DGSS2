import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_sort_pool, GINEConv, global_add_pool


class CellPropagation(nn.Module):
    def __init__(self, node_dim=128, edge_dim=128, dropout_rate=0.5, edge_propagation=True, is_learn=False):
        super(CellPropagation, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.is_learn = is_learn
        self.linear_q = nn.Linear(node_dim, node_dim)
        self.linear_k = nn.Linear(node_dim, node_dim)
        self.linear_v = nn.Linear(node_dim, node_dim)
        self.is_edge_pro = edge_propagation
        if self.is_edge_pro:
            self.edge_linear = nn.Sequential(
                nn.Linear(self.edge_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, self.edge_dim)
            )
        if is_learn:
            self.edge_param = nn.Parameter(torch.randn(128, 1), requires_grad=True)
        self.param = nn.Parameter(torch.randn(100), requires_grad=True)

    def forward(self, graph: Data, cell: torch.Tensor):
        batch = cell.size(0)
        cell = cell.reshape(batch, 1, self.node_dim)
        molecular_x = graph.x.float().reshape(batch, 100, -1)
        mask = graph.mask.reshape(batch, 100)
        if self.is_edge_pro:
            edge_cell = cell.squeeze()
            if self.is_learn:
                edge_cell = self.edge_param * edge_cell
            edge_counts = graph.num_edge
            expand_cell = torch.repeat_interleave(edge_cell, edge_counts, dim=0)
            edge_attr = self.edge_linear(graph.edge_attr) + expand_cell
            graph.edge_attr = edge_attr
        # [128, 100, 78]
        attn_score = self.calc_attn(molecular_x, cell, mask)
        # Propagation
        molecular_x = attn_score + molecular_x
        graph.x = molecular_x.reshape(-1, self.node_dim)
        return graph

    def calc_attn(self, molecular, cell_line, mask):
        query = self.linear_q(molecular)
        key = self.linear_k(cell_line)
        value = self.linear_v(cell_line)
        attn = torch.matmul(query, key.permute(0, 2, 1)).squeeze()
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = (F.softmax(attn, dim=-1) / math.sqrt(self.node_dim)).unsqueeze(-1)
        attn_score = torch.bmm(attn, value)
        # Residuals
        cell_expand = cell_line.expand(cell_line.shape[0], 100, -1)
        cell_expand_mask = ~mask.unsqueeze(2).expand(-1, -1, self.node_dim)
        masked_cell_expand = cell_expand_mask * cell_expand
        param = self.param.reshape(1, -1, 1)
        param = param.expand(cell_line.shape[0], -1, 1)
        out = attn_score + param * masked_cell_expand
        #out = attn_score + masked_cell_expand
        return out


class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False,
                 identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            
            sim = torch.bmm(xn, xn.transpose(1, 2)) / self.temp
            # sim = sim.masked_fill(mask, -1e9)
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            # sim = sim * ~mask
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1 + self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x


class ProcessInteraction(nn.Module):
    def __init__(self, input_dim, hidden_dim, learnable=False):
        super().__init__()
        self.reaction = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.is_learnable = learnable
        if self.is_learnable:
            self.params = nn.Parameter(torch.randn(3, dtype=torch.float32), requires_grad=True)

    def forward(self, graphA: Data, graphB: Data, cell: torch.Tensor):
        batch_size = cell.size(0)
        cell = self.reaction(cell)
        graphA_pool = global_mean_pooling(graphA, batch_size)
        graphB_pool = global_mean_pooling(graphB, batch_size)
        # single drug
        contextA = cell * graphA_pool
        contextB = cell * graphB_pool
        # DDI
        DDI = graphB_pool * graphA_pool
        contextC = DDI * cell
        # Cell Feature (Context)
        if self.is_learnable:
            a, b, c = self.params
            context = a * contextA + b * contextB + c * contextC
        else:
            context = contextA + contextB + contextC
        return context


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)

    def forward(self, x, states=None):
        states = states
        out, (hidden_state, cell_state) = self.lstm(x[None, :, :], states)
        return out, (hidden_state, cell_state)


class MLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, out_dim=128):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def reset_param(self):
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=-0.01, nonlinearity='relu')
                nn.init.uniform_(layer.bias, -1, 0)

        last_layer = self.layers[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.xavier_normal_(last_layer.weight)
            nn.init.uniform_(last_layer.bias, -1, 0)

    def forward(self, x):
        return self.linear(x)


def global_mean_pooling(graph: Data, batch_size):
    drug_features = graph.x.reshape(batch_size, 100, -1)
    mask = ~graph.mask.view(batch_size, 100, -1)
    masked_features = drug_features * mask
    sum_features = torch.sum(masked_features, dim=1)
    valid_counts = torch.sum(mask, dim=1) + 1e-9
    pooled_features = sum_features / valid_counts
    return pooled_features