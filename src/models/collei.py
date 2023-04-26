import torch
import torchmetrics
from src.models.base_model import BaseModelForImageAuthorship
from src.models.blocks import ImageAutorshipEmbeddingBlock
from torch import optim, nn
from torch.nn.init import xavier_uniform_
from src.models.losses import UserwiseAUCROC


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
        super().__init__(d=d, nusers=nusers, lr=lr)
        self.embedding_block = ImageAutorshipEmbeddingBlock(d=d, nusers=nusers)
        self.tau = tau
        self.lr = lr

        xavier_uniform_(self.embedding_block.u_emb.weight.data, gain=1.0)
        xavier_uniform_(self.embedding_block.img_fc.weight.data, gain=1.0)

        self.val_recall = torchmetrics.RetrievalRecall(k=10)
        self.val_auc = UserwiseAUCROC()

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

        users, images, targets, id_tests = batch

        preds = self((users, images))

        self.val_recall.update(preds, targets.long(), id_tests)
        self.val_auc.update(preds, targets.long(), users)

        self.log('val_recall', self.val_recall,
                 on_epoch=True, logger=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_epoch=True,
                 logger=True, prog_bar=True)

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
