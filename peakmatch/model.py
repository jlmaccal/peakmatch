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
        self.gps1 = GPSLayer(dim_h=128, num_heads=4)
        self.gps2 = GPSLayer(dim_h=128, num_heads=4)
        self.readout = ReadoutLayer()

    def forward(self, batch):
        batch = self.init_embed(batch)
        batch = self.gps1(batch)
        batch = self.gps2(batch)
        output = self.readout(batch)
        return output

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)
        loss = 0.0
        for x, y in results:
            loss += cross_entropy(x, y)
        self.log("my_loss", loss,)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss",
                "strict": True,
                "name": None,
                }
            }
    
    # def configure_optimizers(self):
    #     gen_opt = optim.Adam(self.model.parameters(), lr=0.01)
    #     dis_opt = optim.Adam(self.model.parameters(), lr=0.02)
    #     dis_sch = optim.CosineAnnealing(dis_opt, T_max=10)
    #     return [gen_opt, dis_opt], [dis_sch]

