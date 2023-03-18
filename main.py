import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from src.dataset import Tripadvisor_ImageAuthorship_Dataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from src.models.mf_elvis import MF_ELVis
from src.models.elvis import ELVis
from src.test import test_tripadvisor_authorship_task

num_workers = 4

MODE = 'TEST'
MODEL_NAME = 'ELVis'
CITY = 'barcelona'

if __name__ == '__main__':

    dataset = Tripadvisor_ImageAuthorship_Dataset(city=CITY)
    batch_size = 2**15

    if MODEL_NAME == 'MF_ELVis':
        model = MF_ELVis(256, dataset.train_data.nusers)
    elif MODEL_NAME == 'ELVis':
        model = ELVis(256, dataset.train_data.nusers)

    if MODE == 'TRAIN':

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

        trainer.fit(model=model,
                    train_dataloaders=DataLoader(
                        dataset.train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                    val_dataloaders=DataLoader(dataset.val_data, batch_size=batch_size, num_workers=num_workers))

    elif MODE == 'TEST':

        model = model.load_from_checkpoint(
            'models/' + CITY + '/' + MODEL_NAME + '/best-model.ckpt')

        predictor = pl.Trainer(accelerator='gpu', devices=[0])
        predictions = torch.cat(predictor.predict(model, dataloaders=DataLoader(
            dataset.test_data, batch_size=batch_size, num_workers=num_workers)))

        test_tripadvisor_authorship_task(dataset, predictions, MODEL_NAME)
