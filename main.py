import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from src.dataset import ImageAuthorshipDataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from src.models.mf_elvis import MF_ELVis
from src.models.elvis import ELVis
from src.test import test_tripadvisor_authorship_task
from os import remove, path

num_workers = 4

MODE = 'TRAIN'
MODEL_NAME = 'MF_ELVis'
CITY = 'paris'
BATCH_SIZE=2**15

if __name__ == '__main__':

    dm = ImageAuthorshipDataModule(city=CITY, batch_size=BATCH_SIZE)
    dm.setup('initial')

    trainer = pl.Trainer(precision=32, max_epochs=100, accelerator='gpu', devices=[0],
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-3, patience=5),
                                ModelCheckpoint(save_top_k=1,
                                                monitor="val_loss",
                                                mode="min",
                                                dirpath=f"models/{CITY}/{MODEL_NAME}",
                                                filename="best-model",
                                                save_on_train_epoch_end=False
                                                )]

                        )

    
    if MODEL_NAME == 'MF_ELVis':
        model = MF_ELVis(256, dm.nusers)
    elif MODEL_NAME == 'ELVis':
        model = ELVis(256, dm.nusers)

    if MODE == 'TRAIN':

        if path.exists(f'models/{CITY}/{MODEL_NAME}/best-model.ckpt'):
            remove(f'models/{CITY}/{MODEL_NAME}/best-model.ckpt')


        trainer.fit(model=model,train_dataloaders=dm.train_dataloader(),val_dataloaders=dm.val_dataloader())

    elif MODE == 'TEST':
        
        
        model = model.load_from_checkpoint(
            'models/' + CITY + '/' + MODEL_NAME + '/best-model.ckpt')
        
        test_outputs = trainer.test(model=model, datamodule=dm.test_dataloader())
        test_outputs = torch.cat(test_outputs)

        test_tripadvisor_authorship_task(dm, test_outputs, MODEL_NAME)
