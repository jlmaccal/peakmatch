from torch import optim
from torch.nn.functional import cross_entropy
import pytorch_lightning as pl
from .layers.initembed import InitalEmbed
from .layers.readout import ReadoutLayer


class PeakMatchModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.init_embed = InitalEmbed()
        self.readout = ReadoutLayer()

    def forward(self, batch):
        batch = self.init_embed(batch)
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