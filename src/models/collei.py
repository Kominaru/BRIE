import torch
from src.models.base_model import BaseModelForImageAuthorship
from src.models.blocks import ImageAutorshipEmbeddingBlock
from torch import optim, nn


class COLLEI(BaseModelForImageAuthorship):
    """
    Contrastive Loss based method for image autorship perdiction

    Parameters:
        d: int
            Size of the latent image and user embeddings
        nusers: int
            Number of users in the dataset (used for the user embedding layer)
        tau: float
            Temperature of the contrastive loss equation
        lr: float
            Learning rate of the model
    """

    def __init__(self, d: int, nusers: int,  lr: float, tau: float = 1):
        super().__init__(d=d, nusers=nusers)
        self.embedding_block = ImageAutorshipEmbeddingBlock(d=d, nusers=nusers)
        self.tau = tau
        self.lr = lr

    def training_step(self, batch, batch_idx):

        users, images = batch  # Positive samples

        # (batch_size, batch_size)
        preds = self((users, images), is_training=True)

        # Contrastive loss
        loss = -torch.mean(torch.diag(preds))

        # Logging only for print purposes
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        self.train_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):  # See train_step() above

        users, images = batch

        preds = self((users, images), is_training=True)

        # Contrastive loss
        loss = -torch.mean(torch.diag(preds))

        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        self.validation_step_outputs.append(loss)

        return loss

    def forward(self, x, is_training=False):
        user_ids, images = x

        u_embeddings, img_embeddings = self.embedding_block(user_ids, images)

        if is_training:

            preds = u_embeddings @ img_embeddings.T  # (batch_size, batch_size)
            preds = torch.nn.functional.log_softmax(preds/self.tau, dim=0)

        elif not is_training:

            preds = torch.sum(u_embeddings*img_embeddings,
                              dim=-1)  # (batch_size,1)
            preds = torch.sigmoid(preds)

        return preds

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        users, images, targets = batch

        return self((users, images))
