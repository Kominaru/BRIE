import pytorch_lightning as pl
from torch import nn,optim
import torch
import torchmetrics
from src.models.blocks import ImageAutorshipEmbeddingBlock

class MF_ELVis(pl.LightningModule):

    # d: number of latent features to learn from users
    # nusers: number of unique users in the dataset
    # image embedding size is assumed to be 1536

    def __init__(self, d, nusers):
        super().__init__()
        self.embedding_block = ImageAutorshipEmbeddingBlock(d,nusers)


    def training_step(self, batch, batch_idx):
        users, images, targets = batch

        # Lookup user embeddings and obtain reduced image embeddings
        u_embeddings, img_embeddings = self.embedding_block(users,images)

        # Dot product of each sample 
        preds = torch.sum(u_embeddings*img_embeddings, dim=1)
        
        # Using BCEwithLogits for being more numerically stable
        loss = nn.functional.binary_cross_entropy_with_logits(preds, targets)
        accuracy = torchmetrics.functional.accuracy(preds, targets, 'binary')

        # Logging only for print purposes
        self.log_dict({'train_accuracy': accuracy, 'train_loss':loss},on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        return loss

    
    # See train_step() above
    def validation_step(self, batch, batch_idx):

        
        users, images, targets = batch

        u_embeddings, img_embeddings = self.embedding_block(users,images)

        preds = torch.sum(u_embeddings*img_embeddings, dim=1)

        loss = nn.functional.binary_cross_entropy_with_logits(preds, targets)
        accuracy = torchmetrics.functional.accuracy(preds, targets, 'binary')

        self.log_dict({'val_accuracy': accuracy, 'val_loss':loss},on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)

        return loss
    
    # Logging to the Tensorboard dashboard logger
    # https://stackoverflow.com/questions/71236391/pytorch-lightning-print-accuracy-and-loss-at-the-end-of-each-epoch
    def training_epoch_end(self, outputs):
        loss = sum([output['loss'] for output in outputs]) / len(outputs)
        self.logger.experiment.add_scalars('loss', {'train': loss}, self.current_epoch)

    def validation_epoch_end(self, outputs):
        loss = sum(outputs) / len(outputs) 
        self.logger.experiment.add_scalars('loss', {'valid': loss}, self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
