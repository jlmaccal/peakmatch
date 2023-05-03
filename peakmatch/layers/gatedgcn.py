import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
    GatedGCN layer
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        residual,
        activation="relu",
        use_ln=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.use_ln = use_ln

        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "elu":
            self.activation = nn.ELU
        elif activation == "tanh":
            self.activation = nn.Tanh
        else:
            raise ValueError("Invalid activation")

        if self.use_ln:
            self.ln_node_x = nn.LayerNorm(out_dim)
            self.ln_edge_e = nn.LayerNorm(out_dim)
        self.act_fn_x = self.activation()
        self.act_fn_e = self.activation()
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.A.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.B.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.C.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.D.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.E.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.A.bias)
        torch.nn.init.zeros_(self.B.bias)
        torch.nn.init.zeros_(self.C.bias)
        torch.nn.init.zeros_(self.D.bias)
        torch.nn.init.zeros_(self.E.bias)

    def forward(self, batch):
        x = batch.x  # n_nodes x d
        e = batch.edge_attr  # n_edges x d
        edge_index = batch.edge_index  # 2 x n_edges
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax)

        if self.use_ln:
            x = self.ln_node_x(x)
            e = self.ln_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce="sum")

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce="sum")

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out
