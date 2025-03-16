from Layer import *


class GCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.graph_norm = ContraNorm(dim=hidden_dim)

    def forward(self, graph, batch_size):
        edge_index = graph.edge_index
        x = graph.x
        x = F.relu(self.graph_norm(self.gnn(x, edge_index).reshape(batch_size, 100, -1)))
        x = x.reshape(-1, self.hidden_dim)
        graph.x = x
        return graph


class GINELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.BatchNorm1d(2 * input_dim),
            nn.ReLU(),

            nn.Linear(2 * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gnn = GINEConv(self.layers, edge_dim=128)
        self.graph_norm = ContraNorm(dim=hidden_dim)

    def forward(self, graph, batch_size):
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        x = graph.x
        x = F.relu(self.graph_norm(self.gnn(x, edge_index, edge_attr).reshape(batch_size, 100, -1)))
        x = x.reshape(-1, self.hidden_dim)
        graph.x = x
        return graph












