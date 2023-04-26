import pytorch_lightning as pl
from torch import optim, stack


class BaseModelForImageAuthorship(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.lr = kwargs['lr']

    def forward(self, x):
        return NotImplementedError

    def training_step(self, batch, batch_idx):
        return NotImplementedError

    def validation_step(self, batch, batch_idx):
        return NotImplementedError

    # # Logging to the Tensorboard dashboard logger
    # def on_train_epoch_end(self):
    #     training_epoch_loss = stack(self.train_step_outputs).mean()
    #     self.logger.experiment.add_scalars(
    #         'loss', {'train': training_epoch_loss}, self.current_epoch)
    #     self.train_step_outputs.clear()

    # def on_validation_epoch_end(self):
    #     validation_epoch_loss = stack(self.validation_step_outputs).mean()
    #     self.logger.experiment.add_scalars(
    #         'loss', {'valid': validation_epoch_loss}, self.current_epoch)
    #     self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        users, images, _ = batch

        return self((users, images))

    def on_test_epoch_end(self) -> None:
        return

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer