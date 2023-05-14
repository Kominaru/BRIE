import pytorch_lightning as pl
import torch
from src.datamodule import ImageAuthorshipDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from src.test import test_tripadvisor_authorship_task
from os import remove, path
from src.config import read_args
from src import utils
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
import pickle

args = read_args()

if __name__ == "__main__":
    city = args.city
    workers = args.workers

    print("=" * 50)
    print(f"============= {city} ===========")

    val_metric_name = (
        "val_loss" if args.model[0] not in ["PRESLEY", "COLLEI"] else "val_auc"
    )

    val_metric_mode = "min" if args.model[0] not in ["PRESLEY", "COLLEI"] else "max"

    # Initialize datamodule
    dm = ImageAuthorshipDataModule(
        city=city,
        batch_size=args.batch_size,
        num_workers=workers,
        dataset_class=utils.get_dataset_constructor(args.model[0]),
        use_train_val=args.use_train_val,
    )

    # Initialize trainer
    if not args.early_stopping or args.no_validation:
        checkpointing = ModelCheckpoint(
            save_last=True,
            save_top_k=0,
            dirpath=f"models/{city}/{args.model[0]}",
            save_on_train_epoch_end=True,
        )
        callbacks = [checkpointing]
    else:
        early_stopping = EarlyStopping(
            monitor=val_metric_name,
            mode=val_metric_mode,
            min_delta=1e-4,
            patience=10,
            check_on_train_epoch_end=False,
        )

        checkpointing = ModelCheckpoint(
            save_top_k=1,
            monitor=val_metric_name,
            mode=val_metric_mode,
            dirpath=f"models/{city}/{args.model[0]}",
            filename="best-model",
            save_on_train_epoch_end=False,
        )

        callbacks = [checkpointing, early_stopping]

    # Optional CSV logging
    if args.log_to_csv:
        logger = pl.loggers.CSVLogger(
            name=city,
            version=f'{args.model[0]}{"_no_val" if args.no_validation else ""}',
            save_dir="csv_logs",
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        strategy="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger if args.log_to_csv else None,
    )

    ### TRAIN MODE ###
    if args.stage == "train":
        # Initialize model
        model_name = args.model[0]
        model = utils.get_model(model_name, vars(args), dm.nusers)

        # Overwrite model if it already existed
        if path.exists(f"models/{city}/{model_name}/best-model.ckpt"):
            remove(f"models/{city}/{model_name}/best-model.ckpt")
        if path.exists(f"models/{city}/{model_name}/last.ckpt"):
            remove(f"models/{city}/{model_name}/last.ckpt")

        if args.no_validation:
            trainer.fit(model=model, train_dataloaders=dm.train_dataloader())
        else:
            trainer.fit(
                model=model,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(),
            )

    ### HYPERPARAMETER TUNING MODE ###
    if args.stage == "tune":
        model_name = args.model[0]

        # Search space
        config = {
            "lr": tune.loguniform(5e-5, 5e-3),
            "d": tune.choice([4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]),
            # "dropout": tune.choice([0, 0.1, 0.2])
        }

        # Report callback
        tunecallback = TuneReportCallback(
            metrics={
                "val_auc": "val_auc",
                "val_recall": "val_recall",
                "train_loss": "train_loss",
            },
            on=["validation_end", "train_end"],
        )

        # Command line reporter
        reporter = CLIReporter(parameter_columns=["d", "lr", "dropout"])

        # Basic function to train each one
        def train_config(config):
            trainer = pl.Trainer(
                max_epochs=100,
                accelerator="gpu",
                devices=[0],
                callbacks=[tunecallback, early_stopping],
                enable_progress_bar=False,
            )
            model = utils.get_model(model_name, config, nusers=dm.nusers)
            trainer.fit(
                model,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(),
            )

        # Execute analysis
        analysis = tune.run(
            train_config,
            resources_per_trial={"cpu": 16, "gpu": 1},
            metric="val_auc",
            mode="max",
            config=config,
            num_samples=100,
            name=f"tune_{model_name}",
        )

        # Find best configuration and its best val metric value
        best_config = analysis.get_best_config(
            metric=val_metric_name, scope="all", mode=val_metric_mode
        )
        best_val_loss = analysis.dataframe(
            metric=val_metric_name, mode=val_metric_mode
        )[val_metric_name].max()

        print(f"Best {val_metric_name}: {best_val_loss} ({best_config}) ")

    ### TEST/COMPARISON MODE ###
    elif args.stage == "test":
        # Holds predictions of each model to test
        models_preds = {}

        filename = "last" if args.no_validation else "best-model"

        # Obtain predictions of each trained model
        for model_name in args.model:
            if not args.load_preds or model_name in ["PRESLEY"]:
                model = utils.get_model(
                    model_name, vars(args), dm.nusers
                ).load_from_checkpoint(f"models/{city}/{model_name}/{filename}.ckpt")

                test_preds = torch.cat(
                    trainer.predict(model=model, dataloaders=dm.test_dataloader())
                )

            else:
                test_preds = pickle.load(open(f"preds/{city}_{model_name}", "rb"))[
                    "prediction"
                ].values

            models_preds[model_name] = test_preds

        # Obtain random predictions for baseline comparison
        models_preds["RANDOM"] = torch.mean(
            torch.rand((len(dm.test_dataset), 10)), dim=1
        )

        test_tripadvisor_authorship_task(dm, models_preds)
