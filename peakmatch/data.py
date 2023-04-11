import torch
from .laplacian import get_laplacian
from torch_geometric.utils import to_undirected
from collections import namedtuple
import numpy as np
import random


class PeakMatchAugmentedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        residues,
        predicted_contacts,
        missing_peak_frac=0.10,
        extra_peak_frac=0.10,
        hsqc_noise_h=0.2,
        hsqc_noise_n=0.2,
        hsqc_noise_c=0.2,
        noe_frac=1.0,
        noe_threshold=0.05,
    ):
        self.residues = residues
        self.num_residues = self.residues.size(0)
        self.predicted_contacts = to_undirected(predicted_contacts, self.num_residues)
        self.num_predicted_contacts = self.predicted_contacts.size(1)
        self.missing_peak_frac = missing_peak_frac
        self.extra_peak_frac = extra_peak_frac
        self.noe_frac = noe_frac
        self.hsqc_noise_n = hsqc_noise_n
        self.hsqc_noise_h = hsqc_noise_h
        self.hsqc_noise_c = hsqc_noise_c
        self.dummy_res_node = 0
        self.threshold = noe_threshold

    def __iter__(self):
        return self

    def __next__(self):
        residues = self.residues.clone()
        contacts = self.predicted_contacts.clone()

        peak_handler = PeakHandler(
            self.num_residues,
            residues,
            hsqc_noise_n=self.hsqc_noise_n,
            hsqc_noise_h=self.hsqc_noise_n,
            hsqc_noise_c=self.hsqc_noise_c,
        )

        # Sample extra peaks
        extra_peaks = np.random.uniform(0, self.extra_peak_frac)
        missing_peaks = np.random.uniform(0, self.missing_peak_frac)
        n_missing_peaks = np.round(self.num_residues * missing_peaks).astype(int)
        n_extra_peaks = np.round(self.num_residues * extra_peaks).astype(int)

        for i in np.arange(n_missing_peaks):
            peak_handler.remove_peak(i)
        peak_handler.add_noise_peaks(n_extra_peaks)

        # Generate noisifed hsqc
        fake_hsqc = peak_handler.create_fake_hsqc()

        # Create y, will handle missing and extra peaks.
        y = peak_handler.create_y()

        y = y.flatten()

        # handle edges
        noe_edges = peak_handler.create_fake_noe(
            fake_hsqc, contacts, self.noe_frac, self.threshold
        )
        virtual_edges = peak_handler.create_virtual_edges()
        edge_index = torch.cat([contacts, noe_edges, virtual_edges], dim=1)

        num_nodes = peak_handler.total_nodes
        eig_vecs, eig_vals = get_laplacian(edge_index, num_nodes)

        return PeakData(
            res=peak_handler.pred_res_hsqc,
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


class PeakMatchDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=32,
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
        pin_memory_device="",
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
