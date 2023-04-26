from .signnet import MaskedGINDeepSigns
import torch.nn as nn
from torch_geometric.utils import unbatch
from torch_geometric.data import Data, Batch
import torch


class InitalEmbedLayer(nn.Module):
    def __init__(
        self,
        res_dim=3,
        hsqc_dim=3,
        tag_dim=32,
        pos_enc_dim=32,
        embed_dim=128,
        hidden_channels=64,
        out_channels=64,
        num_layers=8,
        rho_num_layers=3,
    ):
        super().__init__()

        self.res_dim = res_dim
        self.hsqc_dim = hsqc_dim
        self.tag_dim = tag_dim
        self.pos_enc_dim = pos_enc_dim
        self.embed_dim = embed_dim

        # Setup layers to embed nodes and edges
        self.embed_res = nn.Linear(
            res_dim, self.embed_dim - self.tag_dim - self.pos_enc_dim
        )
        self.embed_hsqc = nn.Linear(
            self.hsqc_dim, self.embed_dim - self.tag_dim - self.pos_enc_dim
        )

        # Setup tags to identify node and edge types
        self.res_tag = nn.Parameter(torch.randn(1, self.tag_dim))
        self.hsqc_tag = nn.Parameter(torch.randn(1, self.tag_dim))
        self.virtual_tag = nn.Parameter(
            torch.randn(1, self.embed_dim - self.pos_enc_dim)
        )
        self.res_edge_tag = nn.Parameter(torch.randn(1, self.embed_dim))
        self.noe_edge_tag = nn.Parameter(torch.randn(1, self.embed_dim))
        self.virtual_edge_tag = nn.Parameter(torch.randn(1, self.embed_dim))

        # Setup signnet + deep sets
        self.embed_posenc = MaskedGINDeepSigns(
            in_channels=1,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dim_pe=pos_enc_dim,
            rho_num_layers=rho_num_layers,
            use_bn=False,
            dropout=0.0,
            activation="relu",
        )

    def forward(self, batch):
        # We expect a list of PeakData objects.
        # We're going to embed everything and then package it up into a pyg Batch object for
        # later gnn layers.
        node_embeddings = []
        edge_embeddings = []
        raw_edge_index = []
        edge_index = []
        node_count = 0
        n_res = []
        n_hsqc = []
        vecs = []
        batch_indices = []
        ys = []

        for i, data in enumerate(batch):
            ys.append(data.y)

            res_x = self.embed_res(data.res)
            res_x = torch.cat([res_x, self.res_tag.expand(res_x.size(0), -1)], dim=1)

            hsqc_x = self.embed_hsqc(data.hsqc)
            hsqc_x = torch.cat(
                [hsqc_x, self.hsqc_tag.expand(hsqc_x.size(0), -1)], dim=1
            )

            virtual_x = self.virtual_tag.clone()

            res_edge_x = self.res_edge_tag.expand(
                data.contact_edges.size(1), -1
            ).clone()
            noe_edge_x = self.noe_edge_tag.expand(data.noe_edges.size(1), -1).clone()
            virtual_edge_x = self.virtual_edge_tag.expand(
                data.virtual_edges.size(1), -1
            ).clone()

            node_embeddings.append(res_x)
            node_embeddings.append(hsqc_x)
            node_embeddings.append(virtual_x)

            edge_embeddings.append(
                torch.cat([res_edge_x, noe_edge_x, virtual_edge_x], dim=0)
            )

            edge_index.append(data.edge_index + node_count)
            raw_edge_index.append(data.edge_index)

            n_res.append(res_x.size(0))
            n_hsqc.append(hsqc_x.size(0))
            n_nodes = res_x.size(0) + hsqc_x.size(0) + 1
            node_count += n_nodes

            vecs.append(data.eig_vecs)

            batch_indices.append(torch.tensor([i], dtype=torch.long).expand(n_nodes))

        node_embeddings = torch.cat(node_embeddings, dim=0)
        edge_index = torch.cat(edge_index, dim=1)
        vecs = torch.cat(vecs, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)

        # Compute the positional embeddings using sign net
        pos_enc = vecs.unsqueeze(-1)  # N x K x 1
        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0
        pos_enc = self.embed_posenc(
            pos_enc, edge_index, batch_indices
        )  # N x pos_enc_dim
        node_embeddings = torch.cat([node_embeddings, pos_enc], dim=1)  # N x embed_dim

        # Package it all up into a batch
        node_embeddings = unbatch(node_embeddings, batch_indices)
        batch_data = []
        for i, x in enumerate(node_embeddings):
            data = Data(
                x=x, edge_index=raw_edge_index[i], edge_attr=edge_embeddings[i], y=ys[i]
            )
            data.n_res = n_res[i]
            data.n_hsqc = n_hsqc[i]
            batch_data.append(data)
        return Batch.from_data_list(batch_data, follow_batch=["y"])
