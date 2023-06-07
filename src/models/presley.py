import torchmetrics
import torch
from src.models.losses import bpr_loss, UserwiseAUCROC
from src.models.mf_elvis import MF_ELVis
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout


class PRESLEY(MF_ELVis):
    def __init__(self, d: int, nusers: int, lr: float, dropout=0.0):
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

        super().__init__(d=d, nusers=nusers, lr=lr)

        # Dropouts before dot product
        self.user_dropout = Dropout(dropout)
        self.img_dropout = Dropout(dropout)

        xavier_uniform_(self.embedding_block.u_emb.weight.data, gain=1.0)
        xavier_uniform_(self.embedding_block.img_fc.weight.data, gain=1.0)

        self.criterion = None  # Just to sanitize

    def training_step(self, batch, batch_idx):
        users, pos_images, neg_images = batch

        pos_preds = self((users, pos_images), output_logits=True)
        neg_preds = self((users, neg_images), output_logits=True)

        loss = bpr_loss(pos_preds, neg_preds)

        # Logging only for print purposes
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        users, images, targets, id_tests = batch

        preds = self((users, images), output_logits=True)

        self.val_recall.update(preds, targets.long(), id_tests)
        self.val_auc.update(preds, targets.long(), users)

        self.log(
            "val_recall", self.val_recall, on_epoch=True, logger=True, prog_bar=True
        )
        self.log("val_auc", self.val_auc, on_epoch=True, logger=True, prog_bar=True)

    def forward(self, x, output_logits=False):
        users, images = x

        u_embeddings, img_embeddings = self.embedding_block(users, images)

        if output_logits:
            u_embeddings = self.user_dropout(u_embeddings)
            img_embeddings = self.img_dropout(img_embeddings)

        # Using dim=-1 to support forward of batches and single samples
        preds = torch.sum(u_embeddings * img_embeddings, dim=-1)

        if not output_logits:
            preds = torch.sigmoid(preds)
        return preds

    def on_train_epoch_start(self) -> None:
        # Between epochs, resample the negative images of each (user, pos_img, neg_img)
        # sample tryad to avoid overfitting
        self.trainer.train_dataloader.dataset._resample_dataframe()
