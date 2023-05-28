import torch
import pytorch_lightning as pl
from peakmatch import data
from peakmatch.layers import initembed, readout
from peakmatch.model import PeakMatchModel, ModelOptions
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

import mdtraj as md
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.nn.functional import cross_entropy, softmax


def min_max_normalize(data, drange):
    data = np.array(data)
    return (data - drange[0]) / (drange[1] - drange[0])


def mean_normalize(data, drange):
    return (data - np.mean(data)) / (np.max(data) - np.min(data))


def z_score(data):
    return (data - np.mean(data)) / np.std(data)


def transform_data(data, drange, a=-1, b=1):
    return (b - a) * ((data - drange[0]) / (drange[1] - drange[0])) + a


def process_whiten_ucbshift_data(df, hrange, nrange, corange):
    h_shifts = df["H_UCBShift"].to_numpy()
    n_shifts = df["N_UCBShift"].to_numpy()
    co_shifts = df["C_UCBShift"].to_numpy()

    return (
        transform_data(h_shifts, hrange),
        transform_data(n_shifts, nrange),
        transform_data(co_shifts, corange),
    )


if __name__ == "__main__":
    # these are calculated from avg shift values for each residue and atom provided by bmrb
    # hrange = (7.3, 9.3)
    # nrange = (103.3, 129.7)
    # corange = (169.8, 181.8)

    # h_shifts = transform_data(np.array([7.5, 8.5, 9.0]), hrange)
    # n_shifts = transform_data(np.array([105.0, 110.0, 120.0]), nrange)
    # co_shifts = transform_data(np.array([170.0, 175.0, 177.0]), corange)

    # Three points - two of which are indistinguishable from just hsqc
    # n_shifts = np.array([0.0, 0.0, 0.0])
    # h_shifts = np.array([0.0, 0.0, 0.0])
    # co_shifts = np.array([0.0, 0.0, 0.0])

    # x = np.vstack((n_shifts, h_shifts, co_shifts))
    # x = torch.tensor(x).float().T
    # print(x)
    # x = torch.tensor(
    #     [
    #         [-1.0, -1.0, 0.0],
    #         [-1.05, 1.05, 0.0],
    #         [-0.95, 0.95, 0.0],
    #         [1.0, 1.0, 0.0],
    #     ]
    # )

    # e = [(0, 1), (2, 3)]

    N = 40
    x = torch.randn(N, 3)
    e = []
    for i in range(N - 1):
        e.append((i, i + 1))

    batch_size = 32
    params = data.PeakNoiseAndMatchParams()
    dataset = data.PeakMatchAugmentedDataset(
        x,
        e,
        params,
        min_hsqc_completeness=0.7,
        max_hsqc_noise=0.3,
        min_noe_completeness=0.7,
        max_noe_noise=0.3,
    )

    loader = data.PeakMatchDataLoader(dataset, batch_size=batch_size)
    dm = data.PeakMatchDataModule(loader, batch_size=batch_size)

    # options = ModelOptions(dropout=0.0, tag_dim=4, pos_enc_dim=4, embed_dim=16, pos_enc_num_layers=2)
    options = ModelOptions(num_gps_layers=2)
    model = PeakMatchModel(options)
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        limit_train_batches=40,
        logger=tensorboard,
        callbacks=[lr_monitor],
        limit_val_batches=1,
        max_epochs=50,
    )

    trainer.fit(model, datamodule=dm)
