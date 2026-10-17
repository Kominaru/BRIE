"""High-level API to train BRIE on your own data and rank images for a user."""

from __future__ import annotations

import logging
import os
import warnings
from types import SimpleNamespace
from typing import Any, Hashable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader

from brie.datamodule import TripadvisorImageAuthorshipBPRDataset
from brie.models.brie import BRIE

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

ImageFeatures = Union[np.ndarray, pd.DataFrame]


def _as_feature_table(image_features: ImageFeatures) -> tuple[np.ndarray, pd.Index]:
    """Returns a (n_images, n_features) float32 matrix and the id of each of its rows.

    A NumPy array uses the row number as image id; a DataFrame uses its index.
    """
    if isinstance(image_features, pd.DataFrame):
        ids = image_features.index
        matrix = np.array(image_features, dtype=np.float32)
    else:
        matrix = np.asarray(image_features, dtype=np.float32)
        ids = pd.RangeIndex(len(matrix))
    if matrix.ndim != 2:
        raise ValueError(
            f"image_features must be 2-dimensional (n_images, n_features), got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("image_features contains NaN or infinite values")
    if not ids.is_unique:
        raise ValueError("image ids (the index of image_features) must be unique")
    return matrix, ids


class BRIERanker:
    """Personalized ranking of images with BRIE (Bayesian Ranking of Images for Explainability).

    BRIE learns, for every user, a latent representation of the kind of images they
    author, and ranks any set of candidate images (for example, all the photos of a
    restaurant that is being recommended) by how likely they are to be liked by that user.

    The default hyperparameters are the ones used in the paper. Training needs:

    * ``interactions``: one row per image that a user authored, with the user, the image
      and the *item* (e.g. the restaurant) the image belongs to. The item is used to
      sample negative images from the same item, as in the paper.
    * ``image_features``: a pre-computed feature vector per image (the paper uses
      1536-dimensional CNN embeddings, but any size works).

    Example:
        >>> ranker = BRIERanker(d=64).fit(interactions, image_features)
        >>> ranker.rank("alice", candidate_features).head()
        >>> ranker.save("brie.pt")

    Args:
        d: Size of the latent user and image embeddings.
        dropout: Dropout applied to the embeddings during training.
        lr: Learning rate of the Adam optimizer.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        seed: If given, seeds Python, NumPy and PyTorch before training.
        accelerator: PyTorch Lightning accelerator (``"auto"``, ``"cpu"``, ``"gpu"``...).
        num_workers: Data loader workers.
    """

    def __init__(
        self,
        d: int = 64,
        dropout: float = 0.75,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 2**14,
        seed: Optional[int] = None,
        accelerator: str = "auto",
        num_workers: int = 0,
    ) -> None:
        self.d = d
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.accelerator = accelerator
        self.num_workers = num_workers

        self.model: Optional[BRIE] = None
        self.user_ids_: Optional[list] = None
        self._user_index: dict = {}

    # ------------------------------------------------------------------ training

    def fit(
        self,
        interactions: pd.DataFrame,
        image_features: ImageFeatures,
        user_col: str = "user",
        image_col: str = "image",
        item_col: str = "item",
        verbose: bool = True,
    ) -> "BRIERanker":
        """Trains BRIE on (user, image, item) rows and the features of those images.

        Args:
            interactions: One row per image authored by a user.
            image_features: Array of shape ``(n_images, n_features)`` (the row number is
                the image id) or DataFrame indexed by image id. Must contain every image
                in ``interactions``.
            user_col, image_col, item_col: Column names in ``interactions``.
            verbose: Show the training progress bar.
        """
        for col in (user_col, image_col, item_col):
            if col not in interactions.columns:
                raise ValueError(
                    f"Column {col!r} not found in interactions (columns: {list(interactions.columns)})"
                )
        if interactions[[user_col, image_col, item_col]].isna().any().any():
            raise ValueError("interactions contains missing values")

        matrix, image_ids = _as_feature_table(image_features)
        row_of_image = pd.Series(np.arange(len(image_ids)), index=image_ids)
        missing = ~interactions[image_col].isin(row_of_image.index)
        if missing.any():
            examples = interactions.loc[missing, image_col].unique()[:5].tolist()
            raise ValueError(
                f"{int(missing.sum())} interactions refer to images without features, e.g. {examples}"
            )

        users = pd.Index(interactions[user_col].unique()).sort_values()
        if len(users) < 2:
            raise ValueError("At least 2 users are needed to sample negative images")
        users_per_item = interactions.groupby(item_col)[user_col].nunique()
        if not (users_per_item > 1).any():
            raise ValueError(
                "No item has images from 2 or more users, so negative images from the "
                "same item cannot be sampled. Check the item column."
            )

        # The training code works with integer user ids 0..n-1 and image feature rows
        train_df = pd.DataFrame(
            {
                "id_user": users.get_indexer(interactions[user_col]),
                "id_img": row_of_image.loc[interactions[image_col]].to_numpy(),
                "id_restaurant": pd.factorize(interactions[item_col])[0],
                "take": 1,
            }
        )

        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)

        holder = SimpleNamespace(image_embeddings=torch.from_numpy(matrix))
        dataset = TripadvisorImageAuthorshipBPRDataset(
            datamodule=holder,
            city=None,
            partition_name="TRAIN",
            set_type="train",
            dataframe=train_df,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

        model = BRIE(
            d=self.d,
            nusers=len(users),
            lr=self.lr,
            dropout=self.dropout,
            img_dim=matrix.shape[1],
        )
        # Do not write an emissions.csv in the working directory
        model.emissions_tracker = EmissionsTracker(log_level="error", save_to_file=False)

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            devices="auto",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
        )
        with warnings.catch_warnings():
            # Lightning warnings that do not apply here (metrics are logged for the paper's CSV logs)
            warnings.filterwarnings("ignore", message=".*no logger configured.*")
            warnings.filterwarnings("ignore", message=".*does not have many workers.*")
            trainer.fit(model, train_dataloaders=loader)

        self._set_model(model, users.tolist())
        return self

    # ------------------------------------------------------------------ inference

    def score(
        self, users: Union[Hashable, Sequence[Hashable]], image_features: ImageFeatures
    ) -> np.ndarray:
        """Scores in [0, 1] of each image for its user; higher means more likely to be liked.

        Args:
            users: One user (applied to all images) or one user per image.
            image_features: Array ``(n_images, n_features)`` or DataFrame.
        """
        model = self._require_model()
        matrix, _ = _as_feature_table(image_features)
        if matrix.shape[1] != model.embedding_block.img_fc.in_features:
            raise ValueError(
                f"The model expects {model.embedding_block.img_fc.in_features} features per image, "
                f"got {matrix.shape[1]}"
            )
        if isinstance(users, (str, bytes)) or not isinstance(users, (Sequence, np.ndarray, pd.Series, pd.Index)):
            users = [users] * len(matrix)
        if len(users) != len(matrix):
            raise ValueError(f"Got {len(users)} users for {len(matrix)} images")
        user_idx = torch.tensor([self._index_of(u) for u in users], dtype=torch.long)

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            preds = model((user_idx.to(device), torch.from_numpy(matrix).to(device)))
        return preds.cpu().numpy()

    def rank(
        self,
        user: Hashable,
        image_features: ImageFeatures,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Ranks candidate images for one user, best first.

        Args:
            user: A user seen during training.
            image_features: Features of the candidate images (array or DataFrame; the
                row number or the DataFrame index identifies each image in the output).
            top_k: Only return the ``top_k`` best images.

        Returns:
            DataFrame with columns ``image`` and ``score``, sorted by descending score.
        """
        _, image_ids = _as_feature_table(image_features)
        scores = self.score(user, image_features)
        ranking = (
            pd.DataFrame({"image": image_ids, "score": scores})
            .sort_values("score", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        return ranking if top_k is None else ranking.head(top_k)

    @property
    def users(self) -> list:
        """The users the model was trained on."""
        self._require_model()
        return list(self.user_ids_)

    # ------------------------------------------------------------------ persistence

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Saves the model, its hyperparameters and the user ids to a single file."""
        model = self._require_model()
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "hparams": {
                    "d": self.d,
                    "dropout": self.dropout,
                    "lr": self.lr,
                    "img_dim": model.embedding_block.img_fc.in_features,
                },
                "user_ids": self.user_ids_,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike], map_location: Any = "cpu") -> "BRIERanker":
        """Loads a model saved with :meth:`save`."""
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported model file format: {payload.get('format_version')!r}")
        hp = payload["hparams"]
        ranker = cls(d=hp["d"], dropout=hp["dropout"], lr=hp["lr"])
        model = BRIE(
            d=hp["d"],
            nusers=len(payload["user_ids"]),
            lr=hp["lr"],
            dropout=hp["dropout"],
            img_dim=hp["img_dim"],
        )
        model.load_state_dict(payload["state_dict"])
        ranker._set_model(model, list(payload["user_ids"]))
        return ranker

    @classmethod
    def from_lightning_checkpoint(
        cls,
        path: Union[str, os.PathLike],
        user_ids: Optional[Sequence[Hashable]] = None,
        map_location: Any = "cpu",
    ) -> "BRIERanker":
        """Loads a ``.ckpt`` file written by ``main.py`` (including the pre-trained paper models).

        Args:
            path: The checkpoint.
            user_ids: The user behind each embedding row. By default users are the
                integers ``0..n-1``, the ids used in the paper's datasets.
        """
        model = BRIE.load_from_checkpoint(path, map_location=map_location)
        n_users = model.embedding_block.nusers
        if user_ids is None:
            user_ids = list(range(n_users))
        elif len(user_ids) != n_users:
            raise ValueError(f"The checkpoint has {n_users} users, got {len(user_ids)} user_ids")
        ranker = cls(d=model.embedding_block.d)
        ranker._set_model(model, list(user_ids))
        return ranker

    # ------------------------------------------------------------------ internals

    def _set_model(self, model: BRIE, user_ids: list) -> None:
        self.model = model
        self.user_ids_ = user_ids
        self._user_index = {u: i for i, u in enumerate(user_ids)}

    def _require_model(self) -> BRIE:
        if self.model is None:
            raise RuntimeError("The model is not trained: call fit() or load() first")
        return self.model

    def _index_of(self, user: Hashable) -> int:
        try:
            return self._user_index[user]
        except KeyError:
            raise KeyError(
                f"Unknown user {user!r}: BRIE learns an embedding per user, so it can only "
                f"rank images for users seen in training"
            ) from None
