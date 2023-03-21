from src.models.losses import bpr_loss
from src.models.mf_elvis import MF_ELVis


# MF_Elvis adapted to train (and validate) with BPR samples
class PRESLEY(MF_ELVis):

    def __init__(self, d, nusers):
        super().__init__(d, nusers)
        self.criterion = None  # Just to sanitize

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

        users, pos_images, neg_images = batch

        pos_preds = self((users, pos_images), output_logits=True)
        neg_preds = self((users, neg_images), output_logits=True)

        loss = bpr_loss(pos_preds, neg_preds)

        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        self.validation_step_outputs.append(loss)

        return loss
