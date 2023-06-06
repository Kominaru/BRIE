import torch
import torchmetrics
from src.models.base_model import BaseModelForImageAuthorship
from src.models.blocks import ImageAutorshipEmbeddingBlock
from torch.nn.init import xavier_uniform_


class MF_ELVis(BaseModelForImageAuthorship):

    """
    BCE loss based Matrix Factorisation model for image autorship

    Parameters:
        d: int
            Size of the latent image and user embeddings
        nusers: int
            Number of users in the dataset (used for the user embedding layer)
        lr: float
            Learning rate of the model
    """

    def __init__(self, d: int, nusers: int, lr: float):
        super().__init__(d=d, nusers=nusers, lr=lr)

        self.embedding_block = ImageAutorshipEmbeddingBlock(d, nusers)

        self.criterion = torch.nn.BCEWithLogitsLoss()

        xavier_uniform_(self.embedding_block.u_emb.weight.data, gain=1.0)
        xavier_uniform_(self.embedding_block.img_fc.weight.data, gain=1.0)

    def training_step(self, batch, batch_idx):
        users, images, targets = batch
        preds = self((users, images), output_logits=True)

        # Using BCEwithLogits for being more numerically stable
        loss = self.criterion(preds, targets)
        self.train_acc.update(torch.sigmoid(preds), targets)

        # Logging only for print purposes
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    # See train_step() above

    def validation_step(self, batch, batch_idx):
        users, images, targets, id_tests = batch

        preds = self((users, images), output_logits=True)

        loss = self.criterion(preds, targets)
        self.val_acc.update(torch.sigmoid(preds), targets)

        self.val_auc.update(preds, targets.long(), id_tests)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log("val_auc", self.val_auc, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def forward(self, x, output_logits=False):
        users, images = x

        u_embeddings, img_embeddings = self.embedding_block(users, images)

        # Using dim=-1 to support forward of batches and single samples
        preds = torch.sum(u_embeddings * img_embeddings, dim=-1)

        if output_logits:
            return preds
        else:
            preds = torch.sigmoid(preds)
            return preds
