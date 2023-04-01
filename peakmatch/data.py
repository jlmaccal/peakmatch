import torch
from .laplacian import get_laplacian
from torch_geometric.utils import to_undirected
from collections import namedtuple
import numpy as np

PeakData = namedtuple(
    "PeakData",
    "res hsqc contact_edges noe_edges virtual_edges edge_index eig_vecs eig_vals num_nodes y",
)

class PeakHandler():
    def __init__(self, nresidues, pred_res_hsqc, 
                 hsqc_noise_n = 0.0, hsqc_noise_h = 0.0, hsqc_noise_c = 0.0):
        self.nres = nresidues
        self.nhsqc = nresidues
        self.pred_res_hsqc = torch.cat([pred_res_hsqc, torch.zeros(1,3)]) # zeros for dummy residue node
        self.res_indices = torch.arange(0, self.nres).float()
        self.hsqc_noise_n = hsqc_noise_n
        self.hsqc_noise_h = hsqc_noise_h
        self.hsqc_noise_c = hsqc_noise_c 
        self.peak_d = {}
        self._create_d()
        
    @property
    def total_nodes(self):
        self._total_nodes = self.nres + self.nhsqc + 2 # +2 for dummy node and virtual node
        return self._total_nodes
        
    def _create_d(self):
        self.d_hsqc_res = {}
        for res in np.arange(self.nres):
            self.d_hsqc_res[res] = res
            
    def add_noise_peaks(self, n_extra_peaks):
        max_peak_id = np.max(list(self.d_hsqc_res.keys() ))
        for peak_id in np.arange(max_peak_id + 1, max_peak_id + n_extra_peaks + 1):
            self.d_hsqc_res[peak_id] = self.nres
        
        self.nhsqc = self.nhsqc + n_extra_peaks
    
    def remove_peak(self, removed_peak):
        if removed_peak > self.nres:
            raise ValueError(f"{removed_peak} is greater than {self.nres}")
        self.d_hsqc_res[removed_peak] = None
        self.nhsqc = self.nhsqc - 1

    
    def create_fake_hsqc(self):
        fake_hsqc = []
        for peak_id in self.d_hsqc_res.keys():

            if self.d_hsqc_res[peak_id] == None:
                continue

            elif self.d_hsqc_res[peak_id] == self.nres: # The dummy node will be at index self.nres
                # If the peak is an extra peak - we add a bit more noise than usual.
                idx = self.res_indices.multinomial(1)
                # value of 1.2 is arbitrary - could think harder about something else
                sample = torch.normal(self.pred_res_hsqc[idx], 
                                      std = torch.tensor([self.hsqc_noise_n * 1.2, self.hsqc_noise_h * 1.2, self.hsqc_noise_c * 1.2])
                                      )[0] 
                fake_hsqc.append(sample)
            
            else:
                fake_hsqc.append(torch.normal(mean = self.pred_res_hsqc[ self.d_hsqc_res[peak_id] ],
                                              std = torch.tensor([self.hsqc_noise_n, self.hsqc_noise_h, self.hsqc_noise_c])
                                    )
                                )
        
        return torch.stack(fake_hsqc)
                                 
    
    def create_y(self):
        #y = torch.zeros(self.nhsqc, self.nres + 1)
        #assigned_residues = [value for key, value in self.d_hsqc_res.items() if self.d_hsqc_res[key] != None]
        #y[np.arange(0, self.nhsqc), assigned_residues] = 1
        #return y
        return torch.tensor([self.d_hsqc_res[key] for key in self.d_hsqc_res.keys() if self.d_hsqc_res[key] != None])
        
    def create_fake_noe(self, fake_hsqc, contacts, sample_noe=1.0, threshold=0.05):
        
        edges_m = []
        edges_n = []
        base_index = self.nres + 1
        
        self._reindex_peaks()

        for res_i, res_j in zip(contacts[0, :], contacts[1, :]):
            
            peak_i = [key for key, value in self.d_hsqc_res.items() if value == res_i]
            if len(peak_i) == 0:
                continue
            elif len(peak_i) == 1:
                peak_i = peak_i[0]
            else:
                raise ValueError(f"Multiple peaks {peak_i} are assigned to {res_i} from contacts ")
            
            peak_j = [key for key, value in self.d_hsqc_res.items() if value == res_j]
            if len(peak_j) == 0:
                continue
            elif len(peak_j) == 1:
                peak_j = peak_j[0]
            else:
                raise ValueError(f"Multiple peaks {peak_j} are assigned to {res_j} from contacts ")
            
            n1 = fake_hsqc[self.peak_d[peak_i], 0]
            h1 = fake_hsqc[self.peak_d[peak_i], 1]
            h2 = fake_hsqc[self.peak_d[peak_j], 1]
            
            possible1 = torch.argwhere(
                torch.logical_and(
                    (torch.abs(fake_hsqc[:, 0] - n1) < threshold),
                    (torch.abs(fake_hsqc[:, 1] - h1) < threshold),
                )
            )
            possible2 = torch.argwhere(torch.abs(fake_hsqc[:, 1] - h2) < threshold)
            rand_n = torch.rand(1)[0]
            if rand_n <= sample_noe:
                for m in possible1:
                    for n in possible2:
                        edges_m.append(m + base_index)
                        edges_n.append(n + base_index)
            
        return torch.tensor([edges_m, edges_n], dtype=torch.long)
    
    def create_virtual_edges(self):
        edges1 = []
        edges2 = []
        # the virtual node will be at index num_nodes - 1
        for i in range(self.total_nodes - 1):
            edges1.append(i)
            edges2.append(self.total_nodes - 1)
            edges1.append(self.total_nodes - 1)
            edges2.append(i)

        return torch.tensor([edges1, edges2])
    
    def _reindex_peaks(self):
        peak_counter = 0
        for key in self.d_hsqc_res: 
            if self.d_hsqc_res[key] == None:
                continue 
            else:
                self.peak_d[key] = peak_counter
                peak_counter = peak_counter + 1

class PeakMatchAugmentedDataset(torch.utils.data.IterableDataset):
    def __init__(self, residues, predicted_contacts, 
                 missing_peak_frac = 0.10, extra_peak_frac = 0.10, 
                 hsqc_noise_h=0.2, hsqc_noise_n=0.2, hsqc_noise_c=0.2, 
                 noe_frac=1.0, noe_threshold=0.05):
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

        peak_handler = PeakHandler(self.num_residues, residues, 
                                   hsqc_noise_n = self.hsqc_noise_n, 
                                   hsqc_noise_h = self.hsqc_noise_n, 
                                   hsqc_noise_c = self.hsqc_noise_c)

        #Sample extra peaks
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
        noe_edges = peak_handler.create_fake_noe(fake_hsqc, contacts, self.noe_frac, self.threshold)
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
