import torch
from src.models.base_model import BaseModelForImageAuthorship
from src.models.blocks import ImageAutorshipEmbeddingBlock
from torch import optim, nn


class COLLEI(BaseModelForImageAuthorship):

    # d: number of latent features to learn from users
    # nusers: number of unique users in the dataset
    # image embedding size is assumed to be 1536

    def __init__(self, d, nusers, tau=1):
        super().__init__(d, nusers)
        self.embedding_block = ImageAutorshipEmbeddingBlock(d, nusers)
        self.tau = tau

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
