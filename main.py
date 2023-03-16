import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchmetrics
from src.dataset import Tripadvisor_ImageAuthorship_Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.mf_elvis import MF_ELVis

num_workers = 4

MODE = 'TEST'
MODEL_NAME = 'mf_elvis_binary'
CITY = 'barcelona'

if __name__ == '__main__':

    torch.set_float32_matmul_precision('medium')

    dataset = Tripadvisor_ImageAuthorship_Dataset(city=CITY)
    batch_size = 2**15

    model = MF_ELVis(256, dataset.train_data.nusers)

    if MODE == 'TRAIN':


        trainer = pl.Trainer(max_epochs=25, accelerator='gpu', devices=[0], 
                            callbacks=[EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-3, patience=5),
                                       ModelCheckpoint(save_top_k=1,
                                                        monitor="val_loss",
                                                        mode="min",
                                                        dirpath=f"models/{CITY}",
                                                        filename="best-model",
                                                    )]

                            )

        trainer.fit(model=model,
                    train_dataloaders=DataLoader(dataset.train_data, batch_size=batch_size,shuffle=True,num_workers=num_workers),
                    val_dataloaders=DataLoader(dataset.val_data, batch_size=batch_size,num_workers=num_workers))
        
    elif MODE =='TEST':

        predictor = pl.Trainer(accelerator='gpu', devices=[0])
        model = model.load_from_checkpoint('models/'+CITY+'/best-model.ckpt')

        predictions = predictor.predict(model,dataloaders=DataLoader(dataset.test_data, batch_size=batch_size,num_workers=num_workers))
        
        predictions = torch.cat(predictions)
        
        print(torchmetrics.functional.accuracy(predictions, torch.Tensor(dataset.test_data.samples['is_dev']), 'binary'))

