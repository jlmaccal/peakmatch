import torch
import pytorch_lightning as pl
from peakmatch import data
from peakmatch.layers import initembed, readout
from peakmatch.model import PeakMatchModel

x = torch.randn(5, 3).float()
e = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

dataset = data.PeakMatchAugmentedDataset(x, e)
loader = data.PeakMatchDataLoader(dataset, batch_size=32)

model = PeakMatchModel()
trainer = pl.Trainer(limit_train_batches=100)
trainer.fit(model=model, train_dataloaders=loader)