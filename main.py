import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from src.dataset import Tripadvisor_ImageAuthorship_Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.mf_elvis import MF_ELVis

num_workers = 4

if __name__ == '__main__':

    torch.set_float32_matmul_precision('medium')

    dataset = Tripadvisor_ImageAuthorship_Dataset(city='barcelona')
    batch_size = 2**15

    model = MF_ELVis(256, dataset.train_data.nusers)

    trainer = pl.Trainer(max_epochs=25, accelerator='gpu', devices=[0], 
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-3, patience=5)],
                        )

    trainer.fit(model=model,
                train_dataloaders=DataLoader(dataset.train_data, batch_size=batch_size,shuffle=True,num_workers=num_workers),
                val_dataloaders=DataLoader(dataset.val_data, batch_size=batch_size,num_workers=num_workers))
