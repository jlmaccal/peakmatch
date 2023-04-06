import torch
from ..data import PeakHandler

import numpy as np

import unittest


class TestPeakHandler(unittest.TestCase):
    def setUp(self):
        self.N = 5
        self.x = torch.randn(self.N, 3).float()

        e1 = list(range(0, self.N - 1))
        e2 = list(range(1, self.N))
        self.e = torch.tensor([e1, e2], dtype=torch.long)

        self.PeakHandler = PeakHandler(5, self.x, 0.1)
    
    def test_total_nodes_init(self):
        total_nodes = self.N + self.N + 2
        self.assertTrue(total_nodes == self.PeakHandler.total_nodes)
    
    def test_total_nodes_remove_peak(self):
        
        total_nodes = self.N + self.N + 2 - 1
        idx = np.random.randint(0, self.N)
        self.PeakHandler.remove_peak(idx)
        self.assertTrue(total_nodes == self.PeakHandler.total_nodes)
        
        idx = np.random.randint(0, self.N)
        self.PeakHandler.remove_peak(idx)
        total_nodes = self.N + self.N + 2 - 2
        self.assertTrue(total_nodes == self.PeakHandler.total_nodes)
        
    def test_total_nodes_add_peaks(self):
        idx = np.random.randint(0, 10)
        total_nodes = self.N + self.N + 2 + idx
        self.PeakHandler.add_noise_peaks(idx)
        self.assertTrue(total_nodes == self.PeakHandler.total_nodes)
        
    def test_remove_noise_peak_raises_error(self):
        self.PeakHandler.add_noise_peaks(3)
        with self.assertRaises(ValueError):
            self.PeakHandler.remove_peak(self.N + 1)
     
    def test_d_hsqc_res_init(self):
        d = {
            0 : 0,
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4,
        }
        
        self.assertTrue(d == self.PeakHandler.d_hsqc_res)
        
    def test_d_hsqc_res_remove_peak(self):
        # Remove 2
        d = {
            0 : 0,
            1 : 1,
            2 : None,
            3 : 3,
            4 : 4,
        }
        
        self.PeakHandler.remove_peak(2)
        self.assertTrue(d == self.PeakHandler.d_hsqc_res)
        
        # Remove 0
        d = {
            0 : None,
            1 : 1,
            2 : None,
            3 : 3,
            4 : 4,
        }
        
        self.PeakHandler.remove_peak(0)
        self.assertTrue(d == self.PeakHandler.d_hsqc_res)
        
    def test_d_hsqc_res_add_peak(self):
        d = {
            0 : 0,
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4,
            5 : 5,
        }
        
        self.PeakHandler.add_noise_peaks(1)
        self.assertTrue(d == self.PeakHandler.d_hsqc_res)
        
        d = {
            0 : 0,
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4,
            5 : 5,
            6 : 5,
            7 : 5,
        }
        
        self.PeakHandler.add_noise_peaks(2)
        self.assertTrue(d == self.PeakHandler.d_hsqc_res)
    
    def test_create_y_init(self):
        y = torch.zeros(self.N, self.N + 1)
        for i in np.arange(self.N):
            y[i][i] = 1
        self.assertTrue(torch.equal(y, self.PeakHandler.create_y()))
        
    def test_create_y_remove_peak(self):
        y = torch.zeros(self.N, self.N + 1)
        
        for i in np.arange(self.N):
            y[i][i] = 1
            
        remove_idx = np.random.randint(0, 5)
        keep = [i for i in range(y.size(0)) if i != remove_idx]
        y = y[[keep], :].squeeze()
        
        self.PeakHandler.remove_peak(remove_idx)
        
        self.assertTrue(torch.equal(y, self.PeakHandler.create_y()))
        
    def test_create_y_add_noise_peaks(self):
        idx = np.random.randint(0, 5)
        y = torch.zeros(self.N + idx, self.N + 1)
        for i in np.arange(self.N):
            y[i][i] = 1
        for i in np.arange(self.N, self.N + idx):
            y[i][-1] = 1
        
        self.PeakHandler.add_noise_peaks(idx)
        
        
        self.assertTrue(torch.equal(y, self.PeakHandler.create_y()))
        
    def test_create_y_add_noise_peaks_remove_peaks(self):
        n_add = np.random.randint(0, 10)
        
        y = torch.zeros(self.N + n_add, self.N + 1)
        for i in np.arange(self.N):
            y[i][i] = 1
        for i in np.arange(self.N, self.N + n_add):
            y[i][-1] = 1

        remove_idx = np.random.randint(0, 5)
        keep = [i for i in range(y.size(0)) if i != remove_idx]
        y = y[[keep], :].squeeze()

        if remove_idx % 2 == 0:
            self.PeakHandler.add_noise_peaks(n_add)
            self.PeakHandler.remove_peak(remove_idx)
        else:
            self.PeakHandler.remove_peak(remove_idx)
            self.PeakHandler.add_noise_peaks(n_add)        
            
        self.assertTrue(torch.equal(y, self.PeakHandler.create_y()))
        
    def test_fake_hsqc_gen_initial(self):
        self.PeakHandler.hsqc_noise = 0.0
        self.assertTrue(torch.equal(self.x, self.PeakHandler.create_fake_hsqc()))
    
    def test_fake_hsqc_gen_initial_noise(self):
        self.PeakHandler.hsqc_noise = 0.1
        self.assertFalse(torch.equal(self.x, self.PeakHandler.create_fake_hsqc()))
        
    def test_fake_hsqc_add_remove_peaks(self):
        n_add = np.random.randint(0, 10)
        remove_idx = np.random.randint(0, 5)
        
        self.PeakHandler.add_noise_peaks(n_add)
        self.PeakHandler.hsqc_noise = 0.0
        fake_hsqc = self.PeakHandler.create_fake_hsqc()
        
        self.assertTrue(torch.equal(self.x, fake_hsqc[:self.N]))
        self.assertFalse(torch.equal(self.x, fake_hsqc))
        
        self.PeakHandler.remove_peak(remove_idx)
        fake_hsqc = self.PeakHandler.create_fake_hsqc()
        self.assertFalse(torch.equal(self.x, fake_hsqc[:self.N]))
        
    
    def test_create_fake_noe_initial(self):
        fake_hsqc = torch.arange(0, self.N * 3).reshape(self.N, 3)
        base_index = self.N + 1
        noe_edges = self.PeakHandler.create_fake_noe(fake_hsqc, self.e)
        self.assertTrue(torch.equal(self.e, noe_edges - base_index))
        
    def test_create_fake_noe_remove_peak(self):
        e_remove = torch.tensor([[0, 2], [1, 3]])
        base_index = self.N + 1
        
        fake_hsqc = self.PeakHandler.create_fake_hsqc()
       # print(fake_hsqc.shape)
        self.PeakHandler.remove_peak(2)
        fake_hsqc = self.PeakHandler.create_fake_hsqc()
       # print(fake_hsqc.shape)
        
        noe_edges = self.PeakHandler.create_fake_noe(fake_hsqc, self.e)
        print(e_remove, noe_edges - base_index)
        self.assertTrue(torch.equal(e_remove, noe_edges - base_index))


if __name__ == '__main__':
    unittest.main()   