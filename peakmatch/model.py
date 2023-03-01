from torch import optim
from torch.nn.functional import cross_entropy, softmax
import pytorch_lightning as pl
from .layers.initembed import InitalEmbedLayer
from .layers.readout import ReadoutLayer
from .layers.gps import GPSLayer


class PeakMatchModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.init_embed = InitalEmbedLayer()
        self.gps = GPSLayer(dim_h=128, num_heads=4)
        self.readout = ReadoutLayer()

    def forward(self, batch):
        batch = self.init_embed(batch)
        batch = self.gps(batch)
        output = self.readout(batch)
        return output

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)
        loss = 0.0
        for x, y in results:
            loss += cross_entropy(x, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 3e-4)
        return optimizer
