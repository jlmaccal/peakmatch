import numpy as np
import torch

class PeakHandler():
    def __init__(self, nresidues, pred_res_hsqc, hsqc_noise = 0.0):
        self.nres = nresidues
        self.nhsqc = nresidues
        self._total_nodes = self.nres + self.nhsqc + 2 # Plus 2 for dummy residue node and virtual node
        self.pred_res_hsqc = pred_res_hsqc
        self.res_indices = torch.arange(0, self.nres).float()
        self.hsqc_noise = hsqc_noise
        self._create_d()
        
    @property
    def total_nodes(self):
        self._total_nodes = self.nres + self.nhsqc + 2
        return self._total_nodes
        
    def _create_d(self):
        self.d_hsqc_res = {}
        for res in np.arange(self.nres):
            self.d_hsqc_res[res] = res
            
    def add_peaks(self, n_extra_peaks):
        max_peak_id = np.max(self.d_hsqc_res.keys())
        for peak_id in np.arange(max_peak_id, self.max_peak_id + n_extra_peaks):
            self.d_hsqc_res[peak_id] = self.nres
        
        self.nhsqc = self.nhsqc + n_extra_peaks
    
    def remove_peak(self, removed_peak):
        if removed_peak > self.nres:
            raise ValueError(f"{removed_peak} is greater than {self.nres}")
        new_d = {}
        for peak_id, res_id in self.d_hsqc_res.items():
            if peak_id < removed_peak:
                new_d[peak_id] = res_id
            
            elif peak_id == removed_peak:
                continue
            
            else:
                new_d[peak_id - 1] = res_id 
                
        self.d_hsqc_res = new_d
        self.nhsqc = self.nhsqc - 1
    
    def create_fake_hsqc(self):
        fake_hsqc = []
        missing_peaks = []
        for peak_id in self.d_hsqc_res.keys():
            
            if self.d_hsqc_res[peak_id] == self.nres:
                idx = self.res_indices.multinomial(1)
                sample = torch.normal(self.pred_res_hsqc[idx], std=self.hsqc_noise + 0.2)[0] 
                fake_hsqc.append(sample)
            
            else:
                fake_hsqc.append(torch.normal(mean = self.pred_res_hsqc[ self.d_hsqc_res[peak_id] ],
                                              std = self.hsqc_noise
                                    )
                                )
        
        return torch.stack(fake_hsqc), missing_peaks
                                 
                                 
                
    
    def create_y(self):
        return [self.d_hsqc_res[key] for key in self.d_hsqc_res.keys()]
    

    
    def create_fake_noe(self, fake_hsqc, contacts):
        
        edges_m = []
        edges_n = []
        base_index = self.nres + 1
        
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
            
            n1 = fake_hsqc[peak_i, 0]
            h1 = fake_hsqc[peak_i, 1]
            h2 = fake_hsqc[peak_j, 1]
            print(n1, h1, h2)
            
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
    

        