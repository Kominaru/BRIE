import pytorch_lightning as pl
import torch
from src.datamodule import ImageAuthorshipDataModule, TripadvisorImageAuthorshipBPRDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from src.test import test_tripadvisor_authorship_task
from os import remove, path
from src.config import read_args
from src import utils
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

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
                                   patience=5)

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
        model = utils.get_model(model_name, vars(args), dm)

        # Overwrite model if it already existed
        if path.exists(f'models/{args.city}/{model_name}/best-model.ckpt'):
            remove(f'models/{args.city}/{model_name}/best-model.ckpt')

        trainer.fit(model=model, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())

    if args.stage == 'tune':

        model_name = args.model[0]

        # Search space
        config = {
            "lr": tune.loguniform(1e-6, 1e-2),
            "d": tune.lograndint(16, 1536, base=2),
            "batch_size": tune.lograndint(2**6, 2**15, base=2)
        }

        # Report callback
        tunecallback = TuneReportCheckpointCallback(
            metrics={
                "val_loss": "val_loss",
                "train_loss": "train_loss"
            },
            filename="best-model",
            on=["validation_end", "train_end"])
        # Scheduler
        scheduler = ASHAScheduler(
            max_t=100,
            grace_period=3,
            reduction_factor=2)

        # Command line reporter
        reporter = CLIReporter(
            parameter_columns=["d",
                               "lr", "batch_size"],
            metric_columns=["loss", "training_iteration"])

        # Basic function to train each one
        def train_with_hyperparams(config):
            dm = ImageAuthorshipDataModule(
                city=args.city,
                batch_size=config['batch_size'],
                num_workers=args.workers,
                dataset_class=utils.get_dataset_constructor(args.model[0]))
            trainer = pl.Trainer(max_epochs=100, accelerator='gpu', devices=[0],
                                 callbacks=[tunecallback, early_stopping], enable_progress_bar=False)
            model = utils.get_model(model_name, config, nusers=dm.nusers)
            trainer.fit(model, train_dataloaders=dm.train_dataloader(),
                        val_dataloaders=dm.val_dataloader())

        analysis = tune.run(
            train_with_hyperparams,
            scheduler=scheduler,
            resources_per_trial={
                "cpu": args.workers,
                "gpu": 1
            },
            metric="val_loss",
            mode="min",
            config=config,
            num_samples=20,
            local_dir=f"models/{args.city}/{model_name}",
            name=f"tune_{model_name}")

        print(analysis.best_config)

    ### TEST/COMPARISON MODE ###
    elif args.stage == 'test':

        # Holds predictions of each model to test
        models_preds = {}

        for model_name in args.model:

            model = utils.get_model(model_name, args, dm).load_from_checkpoint(
                f'models/{args.city}/{model_name}/best-model.ckpt')

            test_preds = torch.cat(
                trainer.predict(model=model, dataloaders=dm.test_dataloader())
            )

            models_preds[model_name] = test_preds

        models_preds['RANDOM'] = torch.mean(
            torch.rand((len(dm.test_dataset), 10)), dim=1)
        test_tripadvisor_authorship_task(dm, models_preds)
