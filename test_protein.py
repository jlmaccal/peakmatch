import torch
import pytorch_lightning as pl
from peakmatch import data
from peakmatch.layers import initembed, readout
from peakmatch.model import PeakMatchModel

import mdtraj as md
import pandas as pd
import numpy as np
from copy import deepcopy

def min_max_normalize(data, drange):
    data = np.array(data)
    return (data - drange[0] )/(drange[1] - drange[0])

def mean_normalize(data, drange):
    return ( data - np.mean(data) ) / (np.max(data) - np.min(data))

def process_whiten_ucbshift_data(df, hrange, nrange, corange):
    h_shifts = df['H_UCBShift'].to_list()
    n_shifts = df['N_UCBShift'].to_list()
    co_shifts = df['C_UCBShift'].to_list()
    
    return mean_normalize(h_shifts, hrange), mean_normalize(n_shifts, nrange), mean_normalize(co_shifts, corange)

def gen_contacts(pdb_filename):

    # load the structure
    pdb = md.load(pdb_filename)

    # get all of the non-proline residues
    H_indices = []
    for residue in pdb.topology.residues:
        N = list(residue.atoms_by_name("N"))
        H = list(residue.atoms_by_name("H"))
        if not N or not H:
            print(f"Skipped residue {residue}.") # Need to do something else for prolines eventually.
        else:
            N = N[0]
            H = H[0]
            H_indices.append(H.index)
    src = []
    dst = []
    for i, index1 in enumerate(H_indices):
        for j, index2 in enumerate(H_indices):
            if j <= i:
                continue
            xyz1 = pdb.xyz[0, index1, :]
            xyz2 = pdb.xyz[0, index2, :]
            dist = np.linalg.norm(xyz1 - xyz2)
            if dist < 0.5:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)
    
if __name__ == "__main__":
    # completely arbitrary min and maxes. need to figure out more realistic values
    # not doing anything currently.
    hrange=[6.0, 11.0]
    nrange = [95.0, 140.0]
    corange = [168.0, 182.0]

    df = pd.read_csv('ucbshifts_1em7_trunc28.csv')

    h_shifts, n_shifts, co_shifts = process_whiten_ucbshift_data(df, hrange, nrange, corange)

    x = np.vstack((h_shifts, n_shifts, co_shifts)).T
    x = torch.tensor(x).float()

    e = gen_contacts('1em7_protein_G_H_trunc28.pdb')


    dataset = data.PeakMatchAugmentedDataset(x, e, 0.10, 0.1, hsqc_noise=0.1) 
    loader = data.PeakMatchDataLoader(dataset, batch_size=32)

    model = PeakMatchModel()
    trainer = pl.Trainer(limit_train_batches=100, accelerator='gpu', devices=1)
    trainer.fit(model=model, train_dataloaders=loader)
