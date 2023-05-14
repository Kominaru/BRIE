from time import time
import pytorch_lightning as pl
from torch import optim
import torchmetrics
from codecarbon import EmissionsTracker

from src.models.losses import UserwiseAUCROC


class BaseModelForImageAuthorship(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = kwargs["lr"]

        self.val_recall = torchmetrics.RetrievalRecall(k=10)
        self.val_auc = UserwiseAUCROC()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

        self.emissions_tracker = EmissionsTracker(log_level="error")

    def forward(self, x):
        return NotImplementedError

    def training_step(self, batch, batch_idx):
        return NotImplementedError

    def validation_step(self, batch, batch_idx):
        return NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        users, images, _ = batch
        return self((users, images))

    def on_train_epoch_end(self) -> None:
        self.log(
            "carbon_emissions",
            self.emissions_tracker.flush() * 1000,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        # Log elapsed training time
        self.log(
            "time",
            time() - self.emissions_tracker._start_time,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return super().on_train_epoch_end()

    # def on_test_epoch_end(self) -> None:
    #     return

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_fit_start(self) -> None:
        self.emissions_tracker.start()
        return super().on_fit_start()

    def on_fit_end(self) -> None:
        self.emissions_tracker.stop()
        print(
            f"{int(self.emissions_tracker.final_emissions_data.duration//60)}'{int(self.emissions_tracker.final_emissions_data.duration%60)}\" of training time"
        )
        print(
            f"{self.emissions_tracker.final_emissions_data.emissions*1000:.3f}g of CO2"
        )
        print(
            f"{self.emissions_tracker.final_emissions_data.energy_consumed*1000:.3f}Wh of electricity"
        )
        return super().on_fit_end()
