import torch
from GraphBase import *
import warnings
warnings.filterwarnings("ignore")


class NBA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.GraphConvLayers = nn.ModuleList()
        self.GraphConvLayers.append(GINELayer(input_dim=input_dim, hidden_dim=hidden_dim))
        for _ in range(0, num_layers - 1):
            self.GraphConvLayers.append(GINELayer(input_dim=input_dim, hidden_dim=hidden_dim))
        self.CellPropagation = CellPropagation(node_dim=hidden_dim)
        self.lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim)
        self.PInteraction = ProcessInteraction(input_dim=input_dim, hidden_dim=hidden_dim)


    def forward_(self, conv, graph, batch_size, states=None):
        graph = conv(graph, batch_size)
        gcn_hidden = graph.x
        out, (hidden_state, cell_state) = self.lstm(gcn_hidden, states)
        out = out.squeeze()
        return graph, out, (hidden_state, cell_state)

    def forward(self, graphA: Data, graphB: Data, cell: torch.Tensor, batch_size):
        graphA = self.CellPropagation(graphA, cell)
        graphB = self.CellPropagation(graphB, cell)
        statesA, statesB, outA, outB = None, None, None, None
        for conv in self.GraphConvLayers:
            graphA, outA, statesA = self.forward_(conv, graphA, batch_size, statesA)
            graphB, outB, statesB = self.forward_(conv, graphB, batch_size, statesB)
            cell = self.PInteraction(graphA, graphB, cell)
        return graphA, graphB, outA, outB, cell


class CBA(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=3, middle_channels=64, out_dim=2, dropout_rate=0.2):
        super().__init__()
        self.mol_emb = nn.Embedding(512 * 9 + 1, input_dim, padding_idx=0)
        self.edge_emb = nn.Embedding(512 * 3 + 1, input_dim, padding_idx=0)
        self.NBA = NBA(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.reduction = nn.Sequential(
            nn.Linear(908, 2048),  # 908
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim),
            nn.ReLU()
        )
        self.reduction2 = nn.Sequential(
            nn.Linear(908, 2048),  # 908
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim),
            nn.ReLU()
        )
        self.final = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_dim + input_dim, middle_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_channels, out_dim),
        )
        self.graph_transform = MLP()


    def forward(self, graphA: Data, graphB: Data, cell: torch.Tensor):
        batch_size = max(graphA.batch) + 1
        mask1 = graphA.mask.reshape(batch_size, 100)
        mask2 = graphB.mask.reshape(batch_size, 100)

        graphA.x = self.mol_emb(graphA.x).sum(dim=-2)
        graphB.x = self.mol_emb(graphB.x).sum(dim=-2)
        graphA.edge_attr = self.edge_emb(graphA.edge_attr).sum(dim=-2)
        graphB.edge_attr = self.edge_emb(graphB.edge_attr).sum(dim=-2)     
        
        cell = self.reduction2(cell)

        graphA, graphB, outA, outB, cell = self.NBA(graphA, graphB, cell, batch_size)

        graphA_out = self.graph_transform(global_add_pool(graphA.x, graphA.batch))

        graphB_out = self.graph_transform(global_add_pool(graphB.x, graphB.batch))

        outA, outB = global_mean_pool(outA, graphA.batch), global_mean_pool(outB, graphB.batch)
        graph_level = torch.cat([graphA_out, graphB_out], dim=1)
        rnn_level = torch.cat([outA, outB], dim=1)

        out = torch.cat([graph_level, rnn_level, cell], dim=1)

        out = self.final(out)
        return out







