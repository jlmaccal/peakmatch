"""
SignNet https://arxiv.org/abs/2202.13013
based on https://github.com/cptq/SignNet-BasisNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_scatter import scatter


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        use_ln=True,
        dropout=0.0,
        activation="relu",
        residual=False,
    ):
        super().__init__()
        self.lins = nn.ModuleList()
        if use_ln:
            self.lns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_ln:
                    self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation")
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual
        self.reset_params()

    def reset_params(self):
        for layer in self.lins:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_ln:
                x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape:
                x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x


class GIN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        use_ln=True,
        dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        if use_ln:
            self.lns = nn.ModuleList()
        self.use_ln = use_ln
        # input layer
        update_net = MLP(
            in_channels,
            hidden_channels,
            hidden_channels,
            2,
            use_ln=use_ln,
            dropout=dropout,
            activation=activation,
        )
        self.layers.append(GINConv(update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                2,
                use_ln=use_ln,
                dropout=dropout,
                activation=activation,
            )
            self.layers.append(GINConv(update_net))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))
        # output layer
        update_net = MLP(
            hidden_channels,
            hidden_channels,
            out_channels,
            2,
            use_ln=use_ln,
            dropout=dropout,
            activation=activation,
        )
        self.layers.append(GINConv(update_net))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_ln:
                    x = self.lns[i - 1](x)
            x = layer(x, edge_index)
        return x


class MaskedGINDeepSigns(nn.Module):
    """Sign invariant neural network with sum pooling and DeepSet.
    f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dim_pe,
        rho_num_layers,
        use_ln=True,
        dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        self.enc = GIN(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            use_ln=use_ln,
            dropout=dropout,
            activation=activation,
        )
        self.rho = MLP(
            out_channels,
            hidden_channels,
            dim_pe,
            rho_num_layers,
            use_ln=use_ln,
            dropout=dropout,
            activation=activation,
        )
        self.layer_norm = nn.LayerNorm(16)

    def batched_n_nodes(self, batch_index):
        batch_size = batch_index.max().item() + 1
        one = batch_index.new_ones(batch_index.size(0))
        n_nodes = scatter(
            one, batch_index, dim=0, dim_size=batch_size, reduce="add"
        )  # Number of nodes in each graph.
        n_nodes = n_nodes.unsqueeze(1)
        return torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        K = x.shape[1]  # Max. number of eigen vectors / frequencies.

        # Apply layer norm to ensure that the eigenvector components for
        # each node are standardized.
        x = self.layer_norm(x.squeeze(-1)).unsqueeze(-1)

        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out

        batched_num_nodes = self.batched_n_nodes(batch_index)
        mask = torch.cat([torch.arange(K).unsqueeze(0) for _ in range(N)])
        mask = (mask.to(batch_index.device) < batched_num_nodes.unsqueeze(1)).bool()
        x[~mask] = 0

        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe
        return x
