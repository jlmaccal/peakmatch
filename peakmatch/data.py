import torch
from .laplacian import get_laplacian
from torch_geometric.utils import to_undirected
from collections import namedtuple

PeakData = namedtuple(
    "PeakData",
    "res hsqc contact_edges noe_edges virtual_edges edge_index eig_vecs eig_vals num_nodes y",
)


class PeakMatchAugmentedDataset(torch.utils.data.IterableDataset):
    def __init__(self, residues, predicted_contacts, num_fake_hsqc):
        self.residues = residues
        self.num_residues = self.residues.size(0)
        self.predicted_contacts = to_undirected(predicted_contacts, self.num_residues)
        self.num_predicted_contacts = self.predicted_contacts.size(1)
        self.num_fake_hsqc = num_fake_hsqc
        self.dummy_res_node = 0
               
    def __iter__(self):
        return self
    

    def __next__(self):
        residues = self.residues.clone()
        contacts = self.predicted_contacts.clone()

        fake_hsqc = torch.normal(mean=residues, std=0.2)

        if self.num_residues != self.num_fake_hsqc: 
            residues, fake_hsqc = self._handle_peak_mismatch_hsqc(residues, fake_hsqc)
        #num_fake_hsqc = fake_hsqc.size(0)

        noe_edges = _gen_fake_noe(fake_hsqc, contacts, self.num_residues)

        virtual_edges = _gen_virtual_edges(self.num_residues + self.num_fake_hsqc)

        edge_index = torch.cat([contacts, noe_edges, virtual_edges], dim=1)

        num_nodes = self.num_residues + self.num_fake_hsqc + 1 + self.dummy_res_node

        eig_vecs, eig_vals = get_laplacian(edge_index, num_nodes)

        # This will need to be changed when we allow for num_residues != num_fake_hsqc
        #assert self.num_residues == num_fake_hsqc

        if self.num_residues == self.num_fake_hsqc: 
            y = torch.eye(self.num_residues).flatten()
        
        else:
            y = self._handle_peak_mismatch_y()

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
    
    def _handle_peak_mismatch_hsqc(self, residues, fake_hsqc):
        if self.num_residues < self.num_fake_hsqc:
            # If we have more peaks than residues we need to generate additional fake hsqc peaks

            # Get indices of initial fake_hsqc size (identical to residue size)
            indices = torch.arange(0, fake_hsqc.size(0)).float()

            # Sample over those indices with replacement for n_peaks - n_res with replacement
            idx = indices.multinomial(self.num_fake_hsqc - self.num_residues, replacement=True)

            # Add noise to samples to simulate new peaks without a residue explanation
            samples = torch.normal(mean=fake_hsqc[idx], std=0.2) 

            # Combine old fake_hsqc with new samples
            combined_fake_hsqc = torch.cat([fake_hsqc, samples])

            # Add a dummy residue node with zeros for calculated hsqc data and add to residue array
            self.dummy_res_node = 1
            residues = torch.cat([residues, torch.zeros((1, 3))])
            return residues, combined_fake_hsqc
        
        if self.num_residues > self.num_fake_hsqc:
            return residues, fake_hsqc[:self.num_fake_hsqc]
    

    def _handle_peak_mismatch_y(self):
        # If we have more residues than hsqc peaks, that means that some residues do not match with an hsqc peak
        if self.num_residues > self.num_fake_hsqc:
                y = torch.eye(self.num_residues)
                y = y[:, :self.num_fake_hsqc] 
                y = y.flatten()
                return y
        
        # If we have fewer residues than hsqc peaks, we set values of 1 between every unexplainable hsqc peak and our dummy residue node
        elif self.num_residues < self.num_fake_hsqc:
                y = torch.zeros(self.num_residues + self.dummy_res_node, self.num_fake_hsqc)
                res_range = torch.arange(0, self.num_residues)
                y[res_range, res_range] = 1
                y[-1, self.num_residues:] = 1
                y = y.flatten()
                return y

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
