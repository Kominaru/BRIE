from torch import nn, optim
import torch
import torchmetrics
from src.models.base_model import BaseModelForImageAuthorship
from src.models.blocks import ImageAutorshipEmbeddingBlock
from src.models.losses import bpr_loss
from torchmetrics.functional import auroc


class MF_ELVis(BaseModelForImageAuthorship):

    # d: number of latent features to learn from users
    # nusers: number of unique users in the dataset
    # image embedding size is assumed to be 1536

    def __init__(self, d, nusers):
        super().__init__(d, nusers)
        self.embedding_block = ImageAutorshipEmbeddingBlock(d, nusers)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        users, images, targets = batch

        preds = self((users, images), output_logits=True)

        # Using BCEwithLogits for being more numerically stable
        loss = self.criterion(preds, targets)

        # Logging only for print purposes
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        self.train_step_outputs.append(loss)

        return loss

    # See train_step() above

    def validation_step(self, batch, batch_idx):

        users, images, targets = batch

        preds = self((users, images), output_logits=True)

        loss = self.criterion(preds, targets)

        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        self.validation_step_outputs.append(loss)

        return loss

    def forward(self, x, output_logits=False):
        users, images = x

        u_embeddings, img_embeddings = self.embedding_block(users, images)

        # Using dim=-1 to support forward of batches and single samples
        preds = torch.sum(u_embeddings*img_embeddings, dim=-1)

        if output_logits:
            return preds
        else:
            preds = torch.sigmoid(preds)
            return preds

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        users, images, targets = batch

        return self((users, images))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
