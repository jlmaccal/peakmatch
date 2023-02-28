import torch
from .laplacian import get_laplacian
from torch_geometric.utils import to_undirected
from collections import namedtuple

PeakData = namedtuple(
    "PeakData",
    "res hsqc contact_edges noe_edges virtual_edges edge_index eig_vecs eig_vals num_nodes",
)


class PeakMatchAugmentedDataset(torch.utils.data.IterableDataset):
    def __init__(self, residues, predicted_contacts):
        self.residues = residues
        self.num_residues = self.residues.size(0)
        self.predicted_contacts = to_undirected(predicted_contacts, self.num_residues)
        self.num_predicted_contacts = self.predicted_contacts.size(1)

    def __iter__(self):
        return self

    def __next__(self):
        residues = self.residues.clone()
        contacts = self.predicted_contacts.clone()

        fake_hsqc = torch.normal(mean=residues, std=0.2)
        num_fake_hsqc = fake_hsqc.size(0)

        noe_edges = _gen_fake_noe(fake_hsqc, contacts, self.num_residues)

        virtual_edges = _gen_virtual_edges(self.num_residues + num_fake_hsqc)

        edge_index = torch.cat([contacts, noe_edges, virtual_edges], dim=1)

        num_nodes = self.num_residues + num_fake_hsqc + 1

        eig_vecs, eig_vals = get_laplacian(edge_index, num_nodes)

        return PeakData(
            res=residues,
            hsqc=fake_hsqc,
            contact_edges=contacts,
            noe_edges=noe_edges,
            virtual_edges=virtual_edges,
            edge_index=edge_index,
            eig_vecs=eig_vecs,
            eig_vals=eig_vals,
            num_nodes=num_nodes,
        )


def _gen_fake_noe(fake_hsqc, contacts, base_index):
    edges_m = []
    edges_n = []

    for i, j in zip(contacts[0, :], contacts[1, :]):
        n1 = fake_hsqc[i, 0]
        h1 = fake_hsqc[i, 1]
        h2 = fake_hsqc[j, 1]
        possible1 = torch.argwhere(
            torch.logical_and(
                (torch.abs(fake_hsqc[:, 0] - n1) < 0.05),
                (torch.abs(fake_hsqc[:, 1] - h1) < 0.05),
            )
        )
        possible2 = torch.argwhere(torch.abs(fake_hsqc[:, 1] - h2) < 0.05)
        for m in possible1:
            for n in possible2:
                edges_m.append(m + base_index)
                edges_n.append(n + base_index)

    return torch.tensor([edges_m, edges_n], dtype=torch.long)


def _gen_virtual_edges(num_nodes):
    edges1 = []
    edges2 = []
    # the virtual node will be at index num_nodes
    for i in range(num_nodes):
        edges1.append(i)
        edges2.append(num_nodes)
        edges1.append(num_nodes)
        edges2.append(i)

    return torch.tensor([edges1, edges2])


class PeakMatchDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory_device=""
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
