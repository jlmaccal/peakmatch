import torch
from .laplacian import get_laplacian
from torch_geometric.utils import to_undirected
from collections import namedtuple
from .peakhandler import ResidueData
from .peakhandler import PeakNoiseAndMatchParams
from .peakhandler import generate_sample
import pytorch_lightning as pl
import random

PeakData = namedtuple(
    "PeakData",
    "res hsqc contact_edges noe_edges virtual_edges edge_index eig_vecs eig_vals num_nodes y",
)


class PeakMatchAugmentedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        residues,
        predicted_contacts,
        params: PeakNoiseAndMatchParams,
        min_hsqc_completeness=1.0,
        max_hsqc_noise=0.0,
        min_noe_completeness=1.0,
        max_noe_noise=0.0,
    ):
        self.residues = residues
        self.num_residues = self.residues.size(0)
        self.predicted_contacts = predicted_contacts
        self.num_predicted_contacts = len(predicted_contacts)
        self.params = params
        self.min_hsqc_completeness = min_hsqc_completeness
        self.max_hsqc_noise = max_hsqc_noise
        self.min_noe_completeness = min_noe_completeness
        self.max_noe_noise = max_noe_noise

    def __iter__(self):
        return self

    def __next__(self):
        residues = self.residues.clone()
        contacts = self.predicted_contacts.copy()

        res_d = {}
        for residx, shifts in enumerate(residues):
            resdata = ResidueData(
                shift_h=shifts[0],
                shift_n=shifts[1],
                shift_co=shifts[2],
            )

            res_d[residx] = resdata

        peak_handler = generate_sample(
            res_d,
            contacts,
            self.params,
            self.min_hsqc_completeness,
            self.max_hsqc_noise,
            self.min_noe_completeness,
            self.max_noe_noise,
        )

        edge_index = torch.cat(
            [peak_handler.pred_noe, peak_handler.fake_noe, peak_handler.virtual_edges],
            dim=1,
        )
        num_nodes = peak_handler.n_nodes
        eig_vecs, eig_vals = get_laplacian(edge_index, num_nodes)

        return PeakData(
            res=peak_handler.pred_hsqc,
            hsqc=peak_handler.fake_hsqc,
            contact_edges=peak_handler.pred_noe,
            noe_edges=peak_handler.fake_noe,
            virtual_edges=peak_handler.virtual_edges,
            edge_index=edge_index,
            eig_vecs=eig_vecs,
            eig_vals=eig_vals,
            num_nodes=num_nodes,
            y=peak_handler.correspondence,
        )


class CombinedPeakMatchAugmentedDataset(PeakMatchAugmentedDataset):
    def __init__(
        self,
        datasets,
    ):
        self.datasets = datasets

    def __iter__(self):
        return self

    def __next__(self):
        dataset = random.choice(self.datasets)
        return dataset.__next__()


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


class PeakMatchDataModule(pl.LightningDataModule):
    def __init__(self, loader: PeakMatchDataLoader, batch_size: int = 32):
        super().__init__()
        self.loader = loader
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.peak_val = next(iter(self.loader))
        # self.peak_val = self.loader

    def train_dataloader(self):
        return self.loader

    def val_dataloader(self):
        return PeakMatchDataLoader(self.peak_val, batch_size=self.batch_size)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
