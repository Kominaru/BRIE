import torch
import torchmetrics
from src.models.base_model import BaseModelForImageAuthorship
from src.models.blocks import ImageAutorshipEmbeddingBlock
from torch.nn.init import xavier_uniform_
from src.models.losses import bpr_loss
from src.models.mf_elvis import MF_ELVis
from torch.nn import Dropout

class ISLE(MF_ELVis):
    def __init__(self, d: int, lr: float, dropout: float = 0.5):
        """
        BPR (Bayesian Pairwise Ranking) loss based Matrix Factorisation model for image autorship

        Parameters:
            d: int
                Size of the latent image and user embeddings
            nusers: int
                Number of users in the dataset (used for the user embedding layer)
            lr: float
                Learning rate of the model
            dropout: float
                Training dropout of the image and user embeddings before inner product
        """

        super().__init__(d=d, nusers=1, lr=lr)

        # Dropouts before dot product
        self.embedding_block = torch.nn.Linear(1536, d)
        self.user_dropout = Dropout(dropout)
        self.image_dropout = Dropout(dropout)

        xavier_uniform_(self.embedding_block.weight.data, gain=1.0)

        self.criterion = None  # Just to sanitize

    def training_step(self, batch, batch_idx):
        users, masks, pos_images, neg_images = batch

        pos_preds = self((users, masks,pos_images), output_logits=True)
        neg_preds = self((users, masks,neg_images), output_logits=True)

        loss = bpr_loss(pos_preds, neg_preds)

        # Logging only for print purposes
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, users, mask, images, targets, id_tests = batch

        preds = self((users, mask, images), output_logits=True)

        self.val_recall.update(preds, targets.long(), id_tests)
        self.val_auc.update(preds, targets.long(), user_ids)

        self.log(
            "val_recall", self.val_recall, on_epoch=True, logger=True, prog_bar=True
        )
        self.log("val_auc", self.val_auc, on_epoch=True, logger=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        users, masks, images, _ = batch
        return self((users, masks, images))

    def forward(self, x, output_logits=False):
        users, mask, images = x

        users = self.embedding_block(users)
        users = self.user_dropout(users)
        users = users * mask
        users = torch.sum(users, dim=1)
        users = users / torch.sum(mask, dim=1)

        images = self.embedding_block(images)
        images = self.image_dropout(images)

        # Using dim=-1 to support forward of batches and single samples
        preds = torch.sum(users * images, dim=-1)

        if output_logits:
            return preds
        else:
            preds = torch.sigmoid(preds)
            return preds

    def on_train_epoch_start(self) -> None:
        # Between epochs, resample the negative images of each (user, pos_img, neg_img)
        # sample tryad to avoid overfitting
        self.trainer.train_dataloader.dataset._resample_dataframe()
        self.trainer.train_dataloader.dataset._shuffle_sets()
