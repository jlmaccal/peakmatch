import torch
from peakmatch import data
from peakmatch.layers import initembed, readout

x = torch.randn(5, 3).float()
e = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

d = data.PeakMatchAugmentedDataset(x, e)
l = data.PeakMatchDataLoader(d, batch_size=32)

batch = next(iter(l))

layer = initembed.InitalEmbed()
layer2 = readout.ReadoutLayer()
batch = layer(batch)

batch = layer2(batch)

print(batch[0])