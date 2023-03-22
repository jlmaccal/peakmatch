import torch
import pytorch_lightning as pl
from peakmatch import data
from peakmatch.layers import initembed, readout
from peakmatch.model import PeakMatchModel

N = 28
x = torch.randn(N, 3).float()
e1 = list(range(0, N - 1))
e2 = list(range(1, N))
e = torch.tensor([e1, e2], dtype=torch.long)

dataset = data.PeakMatchAugmentedDataset(x, e, 0.1, 0.1, 0.2) 
loader = data.PeakMatchDataLoader(dataset, batch_size=32)

model = PeakMatchModel(dataset.num_residues)
trainer = pl.Trainer(limit_train_batches=100, accelerator='gpu', devices=1)
trainer.fit(model=model, train_dataloaders=loader)
