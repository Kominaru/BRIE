from src.models.losses import bpr_loss
from src.models.mf_elvis import MF_ELVis
from torch import optim, cat
import torchmetrics
from torch.nn.init import xavier_uniform_
from src.models.losses import UserwiseAUCROC


class PRESLEY(MF_ELVis):

    def __init__(self, d: int, nusers: int, lr: float):
        print(d, nusers, lr)
        super().__init__(d=d, nusers=nusers, lr=lr)

        xavier_uniform_(self.embedding_block.u_emb.weight.data, gain=1.0)
        xavier_uniform_(self.embedding_block.img_fc.weight.data, gain=1.0)
        self.criterion = None  # Just to sanitize

        self.val_recall = torchmetrics.RetrievalRecall(k=10)
        self.val_auc = UserwiseAUCROC()

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

        self.val_recall.update(preds, targets.long(), id_tests)
        self.val_auc.update(preds, targets.long(), users)
        self.log('val_recall', self.val_recall,
                 on_epoch=True, logger=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_epoch=True,
                 logger=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.validation_step_outputs.clear()

    def on_train_epoch_start(self) -> None:
        self.trainer.train_dataloader.dataset._resample_dataframe()
