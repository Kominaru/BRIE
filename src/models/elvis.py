from torch import nn, optim
import torch
import torchmetrics
from src.models.base_model import BaseModelForImageAuthorship
from src.models.blocks import ImageAutorshipEmbeddingBlock
from torch.optim.lr_scheduler import CosineAnnealingLR

# ELVis
# by Jorge Díez, Pablo Pérez-Núñez, Oscar Luaces, Beatriz Remeseiro, Antonio Bahamonde
# from "Towards explainable personalized recommendations by learning from users’ photos"
# Information Sciences, Volume 520, 2020, Pages 416-430
# https://doi.org/10.1016/j.ins.2020.02.018


class ELVis(BaseModelForImageAuthorship):

    def __init__(self, d, nusers, lr):
        super().__init__(d=d, nusers=nusers, lr=lr)

        self.embedding_block = ImageAutorshipEmbeddingBlock(d, nusers)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(d*2, d)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(d, 1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')

    def on_train_epoch_start(self):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        users, images, targets = batch

        preds = self((users, images), output_logits=True)

        # Using BCEwithLogits for being more numerically stable
        loss = self.criterion(preds, targets)
        self.train_acc(torch.sigmoid(preds), targets)

        # Logging only for print purposes
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_epoch=True,
                 on_step=False, prog_bar=True)

        self.train_step_outputs.append(loss)

        return loss

    # See training_step() above
    def validation_step(self, batch, batch_idx):

        users, images, targets = batch

        preds = self((users, images), output_logits=True)

        loss = self.criterion(preds, targets)

        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.val_acc(torch.sigmoid(preds), targets)
        self.log('val_acc', self.val_acc, on_epoch=True,
                 on_step=False, prog_bar=True)
        self.validation_step_outputs.append(loss)

        return loss

    def forward(self, x, output_logits=False):
        users, images = x

        # Embedding block
        u_embeddings, img_embeddings = self.embedding_block(users, images)
        concat = torch.cat((u_embeddings, img_embeddings), dim=-1)

        # First MLP block
        concat = torch.relu(concat)
        concat = self.dropout1(concat)
        concat = self.fc1(concat)

        # Second MLP block
        concat = torch.relu(concat)
        concat = self.dropout2(concat)
        concat = self.fc2(concat)

        preds = torch.squeeze(concat)

        # Sigmoid activation for [0,1] preds

        if output_logits:
            return preds
        else:
            preds = torch.sigmoid(preds)
            return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # return optimizer
        scheduler = CosineAnnealingLR(
            optimizer, 100)
        return [optimizer], {'scheduler': scheduler, 'interval': 'epoch'}
