import torch
import pytorch_lightning as pl
from peakmatch import data
from peakmatch.layers import initembed, readout
from peakmatch.model import PeakMatchModel
from pytorch_lightning import loggers as pl_loggers

import mdtraj as md
import pandas as pd
import numpy as np
from copy import deepcopy

def min_max_normalize(data, drange):
    data = np.array(data)
    return (data - drange[0] )/(drange[1] - drange[0])

def mean_normalize(data, drange):
    return ( data - np.mean(data) ) / (np.max(data) - np.min(data))

def z_score(data):
    return ( data - np.mean(data) ) / np.std(data)

def transform_data(data, drange, a = -1, b = 1):
    return (b - a) * ( (data - drange[0]) / (drange[1] - drange[0])  ) + a

def process_whiten_ucbshift_data(df, hrange, nrange, corange):
    h_shifts = df['H_UCBShift'].to_numpy()
    n_shifts = df['N_UCBShift'].to_numpy()
    co_shifts = df['C_UCBShift'].to_numpy()
    
    return transform_data(h_shifts, hrange), transform_data(n_shifts, nrange), transform_data(co_shifts, corange)

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
    pairs = []
    for i, j in zip(src, dst):
        pairs = []
        pairs.append((i, j))
    return pairs
    #return torch.tensor([src, dst], dtype=torch.long)
    
if __name__ == "__main__":
    # these are calculated from avg shift values for each residue and atom provided by bmrb
    hrange = (7.3, 9.3)
    nrange = (103.3, 129.7)
    corange = (169.8, 181.8)

    df = pd.read_csv('ucbshifts_1em7.csv')

    h_shifts, n_shifts, co_shifts = process_whiten_ucbshift_data(df, hrange, nrange, corange)

    x = np.vstack((n_shifts, h_shifts, co_shifts)).T
    x = torch.tensor(x).float()
    #x = torch.randn(x.shape[0], 3).float()

    e = gen_contacts('1em7_protein_G_H.pdb')


    # NOE threshold is currently arbitrary
    batch_size = 32
    dataset = data.PeakMatchAugmentedDataset(
                                            x, 
                                            e,
                                            min_hsqc_completeness=0.8,
                                            max_hsqc_noise=0.1,
                                            min_noe_completeness=0.75,
                                            max_noe_noise=0.1,
                                            ) 
    
    loader = data.PeakMatchDataLoader(dataset, batch_size=batch_size)
    dm = data.PeakMatchDataModule(loader)
    model = PeakMatchModel(batch_size=batch_size)
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="")
    trainer = pl.Trainer(limit_train_batches=100, logger=tensorboard, accelerator='gpu', devices=1)
    trainer.fit(model=model, datamodule=dm, )
