import torch.nn as nn
from torch_geometric.utils import unbatch
from torch.nn.functional import log_softmax
import torch
import math


class ReadoutLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        node_embeddings = unbatch(batch.x, batch.batch)
        ys = unbatch(batch.y, batch.y_batch)

        output = []
        for i, (x, y) in enumerate(zip(node_embeddings, ys)):
            n = batch.n_res[i]
            m = batch.n_hsqc[i]
            residue_embeddings = x[:n, :]
            hsqc_embeddings = x[n : (n + m), :]
            cross_attention = torch.einsum(
                "md,dn->mn", hsqc_embeddings, residue_embeddings.transpose(0, 1)
            ) / math.sqrt(x.size(1))
            cross_attention = log_softmax(cross_attention, dim=1)
            output.append((cross_attention, y))

        return output
