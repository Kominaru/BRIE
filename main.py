import lightning.pytorch as pl
import torch
from src.datamodule import ImageAuthorshipDataModule, TripadvisorImageAuthorshipBPRDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from src.test import test_tripadvisor_authorship_task
from os import remove, path
from src.config import read_args
from src import utils

args = read_args()

if __name__ == '__main__':

    # Initialize datamodule
    dm = ImageAuthorshipDataModule(
        city=args.city,
        batch_size=args.batch_size,
        num_workers=args.workers,
        dataset_class=utils.get_dataset_constructor(args.model[0]))

    # Initialize trainer

    early_stopping = EarlyStopping(monitor="val_loss",
                                   mode="min",
                                   patience=50)

    checkpointing = ModelCheckpoint(save_top_k=1,
                                    monitor="val_loss",
                                    mode="min",
                                    dirpath=f"models/{args.city}/{args.model[0]}",
                                    filename="best-model",
                                    save_on_train_epoch_end=False)

    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', devices=[0],
                         callbacks=[early_stopping, checkpointing])

    ### TRAIN MODE ###
    if args.stage == 'train':

        # Initialize model
        model_name = args.model[0]
        model = utils.get_model(model_name, dm)

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

            model = utils.get_model(model_name, dm).load_from_checkpoint(
                f'models/{args.city}/{model_name}/best-model.ckpt')

            test_preds = torch.cat(
                trainer.predict(model=model, dataloaders=dm.test_dataloader())
            )

            models_preds[model_name] = test_preds

        models_preds['RANDOM'] = torch.mean(
            torch.rand((len(dm.test_dataset), 10)), dim=1)
        test_tripadvisor_authorship_task(dm, models_preds)
