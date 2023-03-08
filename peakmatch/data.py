import torch
from .laplacian import get_laplacian
from torch_geometric.utils import to_undirected
from collections import namedtuple
import numpy as np

PeakData = namedtuple(
    "PeakData",
    "res hsqc contact_edges noe_edges virtual_edges edge_index eig_vecs eig_vals num_nodes y",
)


class PeakMatchAugmentedDataset(torch.utils.data.IterableDataset):
    def __init__(self, residues, predicted_contacts, missing_peak_frac = 0.10, extra_peak_frac = 0.10, hsqc_noise=0.2):
        self.residues = residues
        self.num_residues = self.residues.size(0)
        self.predicted_contacts = to_undirected(predicted_contacts, self.num_residues)
        self.num_predicted_contacts = self.predicted_contacts.size(1)
        self.missing_peak_frac = missing_peak_frac
        self.extra_peak_frac = extra_peak_frac 
        self.hsqc_noise = hsqc_noise 
        self.dummy_res_node = 0
               
    def __iter__(self):
        return self
    

    def __next__(self):
        residues = self.residues.clone()
        contacts = self.predicted_contacts.clone()

        # Sample extra and missing peaks
        fake_hsqc = torch.normal(mean=residues, std=self.hsqc_noise)
        n_extra_peaks = np.round(self.num_residues * self.extra_peak_frac).astype(int)
        n_missing_peaks = np.round(self.num_residues * self.missing_peak_frac).astype(int)

        

        y = torch.eye(self.num_residues)

        # handle effects of additional and missing peaks
        if n_extra_peaks > 0:
            residues, fake_hsqc, y = self._handle_extra_hsqc_peaks(residues, n_extra_peaks, fake_hsqc, y)
            
        if n_missing_peaks > 0:
            fake_hsqc, y = self._handle_missing_hsqc_peaks(n_missing_peaks, fake_hsqc, y)

        # print(f"{fake_hsqc.size(0)}, {n_extra_peaks}, {n_missing_peaks}")
        # print(f"{y.shape}")
        # print(f"{y}")
        # die

        y = y.flatten()

        # handle edges
        noe_edges = _gen_fake_noe(fake_hsqc, contacts, self.num_residues)

        virtual_edges = _gen_virtual_edges(self.num_residues + self.dummy_res_node + fake_hsqc.size(0))

        edge_index = torch.cat([contacts, noe_edges, virtual_edges], dim=1)

        num_nodes = self.num_residues + self.dummy_res_node + 1 + fake_hsqc.size(0)

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
            y=y,
        )
    
    def _handle_extra_hsqc_peaks(self, residues, n_extra_peaks, fake_hsqc, y):

        # Get indices of residues size
        indices = torch.arange(0, self.num_residues).float()

        # Sample over those indices with replacement for n_extra_peaks
        idx = indices.multinomial(n_extra_peaks, replacement=True)

        # Add noise to samples to simulate new peaks without an associated residue 
        # we could use residues or fake_hsqc here, using fake_hsqc makes extra peaks noisier.
        samples = torch.normal(mean=fake_hsqc[idx], std=0.2) 

        # Combine old fake_hsqc with new samples
        combined_fake_hsqc = torch.cat([fake_hsqc, samples])

        # Add a dummy residue node with zeros for calculated hsqc data and add to residue array
        self.dummy_res_node = 1
        residues = torch.cat([residues, torch.zeros((1, 3))])

        ## y also has to change to reflect extra peaks added ##

        # Add additional hsqc columns to y
        y = torch.hstack((y, torch.zeros(y.size(0), n_extra_peaks)))

        assert y.size(1) == combined_fake_hsqc.size(0)

        # Add row to y for dummy residue node
        y = torch.cat((y, torch.zeros(1, combined_fake_hsqc.size(0))))

        # Set dummy residue node attention to fake hsqc peaks to 1.
        y[-1, self.num_residues:] = 1

        return residues, combined_fake_hsqc, y

    def _handle_missing_hsqc_peaks(self, n_missing_peaks, fake_hsqc, y):
        # Get indices of residues size
        # Sample only up to original residue size so that we do not mess with the dummy residue or extra peaks that may have been added
        indices = torch.arange(0, self.num_residues).float()

        # Sample over those indices with replacement for n_missing_peaks
        # set replacement to False since we want the randomly chosen missing peaks to be unique
        idx = indices.multinomial(n_missing_peaks, replacement=False)
        keep = torch.ones(fake_hsqc.size(0), dtype=bool)
        keep[idx] = False

        # Index fake_hsqc to only grab kept peaks
        reduced_fake_hsqc = fake_hsqc[keep]

        ## y also has to change to reflect missing peaks ##
        y = y[:, keep]

        assert y.size(1) == reduced_fake_hsqc.size(0)

        return reduced_fake_hsqc, y


def _gen_fake_noe(fake_hsqc, contacts, base_index):
    edges_m = []
    edges_n = []

    for i, j in zip(contacts[0, :], contacts[1, :]):
        # If statement to handle when there are fewer hsqc peaks than residues
        if i < fake_hsqc.size(0) and j < fake_hsqc.size(0):
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
