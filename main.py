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
from src.config import read_args


def get_model(model_name, dm):
    if model_name == 'MF_ELVis':
        model = MF_ELVis(256, dm.nusers)
    elif model_name == 'ELVis':
        model = ELVis(256, dm.nusers)
    return model


args = read_args()

num_workers = 4

BATCH_SIZE = 2**15

if __name__ == '__main__':

    # Initialize dataset
    dm = ImageAuthorshipDataModule(city=args.city, batch_size=BATCH_SIZE)
    dm.setup('initial')

    # Initialize trainer object
    trainer = pl.Trainer(precision=32, max_epochs=100, accelerator='gpu', devices=[0],
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-3, patience=5),
                                    ModelCheckpoint(save_top_k=1,
                                                    monitor="val_loss",
                                                    mode="min",
                                                    dirpath=f"models/{args.city}/{args.model[0]}",
                                                    filename="best-model",
                                                    save_on_train_epoch_end=False
                                                    )]

                         )

    ### TRAIN MODE ###
    if args.stage == 'train':

        # Initialize model
        model_name = args.model[0]
        model = get_model(model_name, dm)

        # Overwrite model if it already existed
        if path.exists(f'models/{args.city}/{model_name}/best-model.ckpt'):
            remove(f'models/{args.city}/{model_name}/best-model.ckpt')

        trainer.fit(model=model, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())

    ### TEST/COMPARISON MODE ###
    elif args.stage == 'test':

        # Holds predictions of each model to test
        models_preds = {}

        for model_name in args.model:

            model = get_model(model_name, dm)

            model = model.load_from_checkpoint(
                f'models/{args.city}/{model_name}/best-model.ckpt')

            test_preds = trainer.predict(
                model=model, dataloaders=dm.test_dataloader())

            test_preds = torch.cat(test_preds)

            models_preds[model_name] = test_preds

        test_tripadvisor_authorship_task(dm, models_preds)
