import torch
from torch import optim
from torch.nn.functional import cross_entropy, softmax
import pytorch_lightning as pl
import torchmetrics
from .layers.initembed import InitalEmbedLayer
from .layers.readout import ReadoutLayer
from .layers.gps import GPSLayer
from torchmetrics.functional.classification import multiclass_accuracy
from .plots import *


class PeakMatchModel(pl.LightningModule):
    def __init__(self, batch_size, layer_norm=True):
        super().__init__()
        self.init_embed = InitalEmbedLayer(use_ln=layer_norm)
        self.gps1 = GPSLayer(dim_h=128, num_heads=4, layer_norm=layer_norm)
        self.gps2 = GPSLayer(dim_h=128, num_heads=4, layer_norm=layer_norm)
        self.readout = ReadoutLayer()
        self.batch_size = batch_size

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
            accuracy += multiclass_accuracy(
                x, y, num_classes=x.size(1), average="micro"
            )

        loss = loss / self.batch_size
        accuracy = accuracy / self.batch_size

        self.log_dict(
            {"loss": loss, "accuracy": accuracy},
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return {
            "loss": loss,
            "accuracy": accuracy,
            "x": x,
            "y": y,
        }

    def validation_step(self, batch, batch_idx):
        results = self.forward(batch)
        loss = 0.0
        accuracy = 0.0
        for x, y in results:
            print()
            print()
            print(x.detach())
            print()
            print()
            loss += cross_entropy(x, y)
            accuracy += multiclass_accuracy(
                x, y, num_classes=x.size(1), average="micro"
            )

        loss = loss / self.batch_size
        accuracy = accuracy / self.batch_size

        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy},
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return {
            "val_loss": loss,
            "val_accuracy": accuracy,
            "val_x": x,
            "val_y": y,
        }
        results = self.forward(batch)
        loss = 0.0
        accuracy = 0.0
        figs = []
        # entropy_figs = []
        for idx, (x, y) in enumerate(results):
            loss += cross_entropy(x, y)
            accuracy += multiclass_accuracy(
                x, y, num_classes=x.size(1), average="micro"
            )
        #     figs.append(gen_assignment_fig(batch[idx].res.cpu(),
        #                                    batch[idx].hsqc.cpu(),
        #                                    x.cpu(),
        #                                    y.cpu(),
        #                                    )
        #                                 )
        #     entropy_figs.append(plot_entropy_hsqc(
        #                                     batch[idx].res.cpu(),
        #                                     x.cpu(),

        #                                     )
        #                                 )
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy / self.batch_size},
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        # self.logger.experiment.add_figure('assign', figs)
        # self.logger.experiment.add_figure('entropy', entropy_figs)
        return loss, accuracy

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 3e-4)
        # scheduler1 = optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=1e-2, total_iters=5
        # )
        # scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        # scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "epoch",
            #     "frequency": 1,
            #     "monitor": "loss",
            #     "strict": True,
            #     "name": None,
            # },
        }
