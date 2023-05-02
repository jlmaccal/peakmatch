import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from .gatedgcn import GatedGCNLayer


class GPSLayer(nn.Module):
    def __init__(
        self,
        dim_h,
        num_heads,
        activation="relu",
        dropout=0.0,
        layer_norm=True,
    ):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.local_gnn_type = "gatedgcn"
        self.global_model_type = "graphgps"
        self.layer_norm = layer_norm
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "elu":
            self.activation = nn.ELU
        elif activation == "tanh":
            self.activation = nn.Tanh
        else:
            raise ValueError("Invalid activation")

        # Local message-passing model.
        self.local_model = GatedGCNLayer(
            dim_h,
            dim_h,
            dropout=dropout,
            residual=True,
            act=activation,
            use_ln=self.layer_norm,
        )

        # global attention
        self.self_attn = torch.nn.MultiheadAttention(
            dim_h, num_heads, dropout=dropout, batch_first=True
        )

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.

            local_out = self.local_model(
                Batch(
                    batch=batch,
                    x=h,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                )
            )
            # GatedGCN does residual connection and dropout internally.
            h_local = local_out.x
            batch.edge_attr = local_out.edge_attr
            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            h_attn = self._sa_block(h_dense, None, ~mask)[mask]

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = (
            f"summary: dim_h={self.dim_h}, "
            f"local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, "
            f"heads={self.num_heads}"
        )
        return s
