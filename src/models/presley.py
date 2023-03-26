from src.models.losses import bpr_loss
from src.models.mf_elvis import MF_ELVis
from torch import optim, cat
import torchmetrics


class PRESLEY(MF_ELVis):

    def __init__(self, d: int, nusers: int, lr: float):
        print(d, nusers, lr)
        super().__init__(d=d, nusers=nusers, lr=lr)
        self.criterion = None  # Just to sanitize

        self.validation_step_preds = []
        self.validation_step_targets = []
        self.validation_step_indexes = []

    def training_step(self, batch, batch_idx):
        users, pos_images, neg_images = batch

        pos_preds = self((users, pos_images), output_logits=True)
        neg_preds = self((users, neg_images), output_logits=True)

        loss = bpr_loss(pos_preds, neg_preds)

        # Logging only for print purposes
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        self.train_step_outputs.append(loss)

        return loss

    # See train_step() above

    def validation_step(self, batch, batch_idx):

        users, images, targets, id_tests = batch

        preds = self((users, images), output_logits=True)

        self.validation_step_preds.append(preds)
        self.validation_step_targets.append(targets)
        self.validation_step_indexes.append(id_tests)

    def on_validation_epoch_end(self):
        preds = cat(self.validation_step_preds)
        targets = cat(self.validation_step_targets)
        indexes = cat(self.validation_step_indexes)

        recall_at_10 = torchmetrics.RetrievalRecall(k=10)(
            preds=preds, target=targets, indexes=indexes)
        self.log('val_loss', 1-recall_at_10,
                 prog_bar=True, logger=True, on_epoch=True)
        self.validation_step_outputs.clear()
        self.validation_step_preds.clear()
        self.validation_step_targets.clear()
        self.validation_step_indexes.clear()

    def on_train_epoch_start(self) -> None:
        self.trainer.train_dataloader.dataset._resample_dataframe()
