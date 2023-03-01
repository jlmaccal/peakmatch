import torch
import pytorch_lightning as pl
from peakmatch import data
from peakmatch.layers import initembed, readout
from peakmatch.model import PeakMatchModel

N = 4
x = torch.randn(N, 3).float()
e1 = list(range(0, N - 1))
e2 = list(range(1, N))
e = torch.tensor([e1, e2], dtype=torch.long)

dataset = data.PeakMatchAugmentedDataset(x, e)
loader = data.PeakMatchDataLoader(dataset, batch_size=32)

model = PeakMatchModel()
trainer = pl.Trainer(limit_train_batches=100)
trainer.fit(model=model, train_dataloaders=loader)
