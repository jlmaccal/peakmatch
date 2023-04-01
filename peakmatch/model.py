import torch
from torch import optim
from torch.nn.functional import cross_entropy, softmax
import pytorch_lightning as pl
import torchmetrics
from .layers.initembed import InitalEmbedLayer
from .layers.readout import ReadoutLayer
from .layers.gps import GPSLayer



class PeakMatchModel(pl.LightningModule):
    def __init__(self, nres):
        super().__init__()

        nclasses = nres + 1
        self.init_embed = InitalEmbedLayer()
        self.gps1 = GPSLayer(dim_h=128, num_heads=4)
        self.gps2 = GPSLayer(dim_h=128, num_heads=4)
        self.readout = ReadoutLayer()
        #self.accuracy = torchmetrics.Accuracy(task = 'binary', threshold = 1e-2, multidim_average='global')
        self.accuracy = torchmetrics.Accuracy(num_classes=nclasses, task = 'multiclass', multidim_average='global')

    def forward(self, batch):
        batch = self.init_embed(batch)
        batch = self.gps1(batch)
        batch = self.gps2(batch)
        output = self.readout(batch)
        return output

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)
        loss = 0.0
        accuracy = 0.0
        for x, y in results:
            loss += cross_entropy(x, y)
           # accuracy += self.accuracy(torch.argmax(x, dim=0), torch.argmax(y, dim=0))
            accuracy += self.accuracy(x, y)
    
        self.log_dict({"loss": loss, "multiclass_accuracy": accuracy / len(batch)}, on_step=True, on_epoch=True, prog_bar=False, batch_size = len(batch) )
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

