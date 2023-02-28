import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix
from torch_geometric.utils import get_laplacian as pyg_get_laplacian
import numpy as np


def get_laplacian(
    edge_index, num_nodes, laplace_norm_type=None, eigvec_norm_type="L2", max_freqs=16
):
    undir_edge_index = to_undirected(edge_index)

    L = to_scipy_sparse_matrix(
        *pyg_get_laplacian(
            undir_edge_index, normalization=laplace_norm_type, num_nodes=num_nodes
        )
    )
    vals, vecs = np.linalg.eigh(L.toarray())
    vals, vecs = get_lap_decomp_stats(
        evals=vals, evects=vecs, max_freqs=max_freqs, eigvec_norm=eigvec_norm_type
    )

    return vecs, vals


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm="L2"):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.
    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        vecs = F.pad(evects, (0, max_freqs - N), value=float("nan"))
    else:
        vecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        vals = F.pad(evals, (0, max_freqs - N), value=float("nan")).unsqueeze(0)
    else:
        vals = evals.unsqueeze(0)
    vals = vals.repeat(N, 1).unsqueeze(2)

    return vals, vecs


def eigvec_normalizer(vecs, vals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    vals = vals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = vecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = vecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(vecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(vecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(vals)
        eigval_denom[vals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = (
            torch.max(vecs.abs(), dim=0, keepdim=True)
            .values.clamp_min(eps)
            .expand_as(vecs)
        )
        vecs = torch.asin(vecs / denom_temp)
        eigval_denom = torch.sqrt(vals)
        eigval_denom[vals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(vecs.abs(), dim=0) * vecs.abs()).sum(
            dim=0, keepdim=True
        )
        eigval_denom = torch.sqrt(vals)
        eigval_denom[vals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(vecs)
    vecs = vecs / denom

    return vecs
