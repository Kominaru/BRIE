import pickle
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from pytorch_lightning import LightningDataModule
import pandas as pd
from numpy.random import randint
import numpy as np


# City-wise Datamodule, contains the image embeddings (common to all partitions)
# and all the required partitions (train, train+val, val, test)


class ImageAuthorshipDataModule(LightningDataModule):
    def __init__(
        self, city, batch_size, num_workers=4, dataset_class=None, use_train_val=False
    ) -> None:
        super().__init__()
        self.city = city
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_class = (
            TripadvisorImageAuthorshipBCEDataset
            if dataset_class is None
            else dataset_class
        )
        self.use_train_val = use_train_val

        self.setup()

    def setup(self, stage=None):
        self.image_embeddings = Tensor(
            pickle.load(
                open(
                    "C:/Users/Komi/Papers/BRIE/data/"
                    + self.city
                    + "/data_10+10/IMG_VEC",
                    "rb",
                )
            )
        )

        self.train_dataset = self._get_dataset(
            "TRAIN" if not self.use_train_val else "TRAIN_DEV"
        )
        self.train_val_dataset = self._get_dataset("TRAIN_DEV")
        self.val_dataset = self._get_dataset(
            "DEV" if not self.use_train_val else "TEST", set_type="validation"
        )
        self.test_dataset = self._get_dataset("TEST", set_type="test")

        print(
            f"{self.city:<10} | {self.train_dataset.nusers} users | {len(self.image_embeddings)} images"
        )

        self.nusers = self.train_dataset.nusers

    def _get_dataset(self, set_name, set_type="train"):
        return self.dataset_class(
            datamodule=self, city=self.city, partition_name=set_name, set_type=set_type
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


# Dataset to train with BCE criterion
# Compatible with models: MF_ELVis,  ELVis
class TripadvisorImageAuthorshipBCEDataset(Dataset):
    def __init__(
        self,
        datamodule: ImageAuthorshipDataModule,
        city=None,
        partition_name=None,
        set_type="train",
    ):
        self.set_type = set_type
        self.city = city
        self.datamodule = datamodule
        self.partition_name = partition_name

        # Name of the column that indicates sample label varies between partitions
        self.takeordev = "is_dev" if partition_name in ["DEV", "TEST"] else "take"

        self.dataframe = pickle.load(
            open(
                f"C:/Users/Komi/Papers/BRIE/data/{city}/data_10+10/{partition_name}_IMG",
                "rb",
            )
        )

        self.nusers = self.dataframe["id_user"].nunique()

        print(
            f"{self.set_type} partition ({self.partition_name}_IMG)   | {len(self.dataframe)} samples | {self.nusers} users"
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = self.dataframe.at[idx, "id_user"]
        image_id = self.dataframe.at[idx, "id_img"]
        image = self.datamodule.image_embeddings[image_id]

        target = float(self.dataframe.at[idx, self.takeordev])

        if self.set_type == "train" or self.set_type == "test":
            return user_id, image, target

        elif self.set_type == "validation":
            id_test = self.dataframe.at[idx, "id_test"]
            return user_id, image, target, id_test


# Dataset to train with BPR criterion
# Compatible with models: PRESLEY
class TripadvisorImageAuthorshipBPRDataset(TripadvisorImageAuthorshipBCEDataset):
    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipBPRDataset, self).__init__(**kwargs)
        if self.set_type == "train":
            self._setup_bpr_dataframe()

    # Creates the BPR criterion samples with the form (user, positive image, negative image)
    def _setup_bpr_dataframe(self):
        # Separate between positive and negative samples
        self.positive_samples = (
            self.dataframe[self.dataframe[self.takeordev] == 1]
            .sort_values(["id_user", "id_img"])
            .rename(columns={"id_img": "id_pos_img"})
            .drop_duplicates(keep="first")
            .reset_index(drop=True)
        )

        self.user_centroids = np.stack(
            self.positive_samples.groupby("id_user")
            .apply(
                lambda x: self.datamodule.image_embeddings[x["id_pos_img"].values].mean(
                    axis=0
                )
            )
            .values
        )

        self.p90 = (
            self.positive_samples.groupby("id_user")
            .apply(
                lambda x: np.percentile(
                    np.linalg.norm(
                        self.datamodule.image_embeddings[x["id_pos_img"].values]
                        - self.user_centroids[x.name],
                        axis=1,
                        ord=2,
                    ),
                    90,
                )
            )
            .values
        )

        self._resample_dataframe()

    def _resample_dataframe(self):

        user_ids = self.positive_samples["id_user"].values
        img_ids = self.positive_samples["id_pos_img"].values

        neg_ids = []

        for _ in range(40):

            rand_negs = np.random.randint(len(img_ids), size=len(img_ids))

            invalid_negs = np.arange(img_ids.shape[0])

            i = 0
            max_iter = 5
            while len(invalid_negs) > 0:
                rand_negs[invalid_negs] = np.random.randint(
                    len(img_ids), size=len(invalid_negs)
                )
                invalid_dist = np.where(
                    np.linalg.norm(
                        self.datamodule.image_embeddings[img_ids[rand_negs]]
                        - self.user_centroids[user_ids],
                        axis=1,
                        ord=2,
                    )
                    < self.p90[user_ids]
                )[0]
                invalid_same = np.where(user_ids == user_ids[rand_negs])[0]
                if i < max_iter:
                    invalid_negs = np.union1d(invalid_dist, invalid_same)
                else:
                    invalid_negs = invalid_same
                i += 1

            neg_ids.append(img_ids[rand_negs])

        def obtain_samerest_samples(rest):
            # Works the same way as the previous algorithm, but only restaurant-wise
            user_ids = rest["id_user"].values
            img_ids = rest["id_pos_img"].values

            rand_negs = np.random.randint(len(img_ids), size=len(img_ids))

            invalid_negs = np.arange(img_ids.shape[0])

            i = 0
            while len(invalid_negs) > 0 and i < 0:
                rand_negs[invalid_negs] = np.random.randint(
                    len(img_ids), size=len(invalid_negs)
                )
                invalid_negs = np.where(
                    (
                        np.linalg.norm(
                            self.datamodule.image_embeddings[img_ids[rand_negs]]
                            - self.user_centroids[user_ids],
                            axis=1,
                            ord=2,
                        )
                        < self.p90[user_ids]
                    )
                    | (user_ids == user_ids[rand_negs])
                )[0]
                i += 1
            rest["id_neg_img"] = img_ids[rand_negs]

            return rest

        # twenty_duplicates = pd.concat([self.positive_samples] * 20, ignore_index=True)

        # twenty_duplicates = (
        #     twenty_duplicates.groupby("id_restaurant")
        #     .filter(lambda g: g["id_user"].nunique() > 1)
        #     .reset_index(drop=True)
        # )

        # twenty_duplicates = (
        #     twenty_duplicates.groupby("id_restaurant", group_keys=False)
        #     .apply(obtain_samerest_samples)
        #     .reset_index(drop=True)
        # )

        self.bpr_dataframe = pd.concat([self.positive_samples] * 40, ignore_index=True)
        self.bpr_dataframe["id_neg_img"] = np.concatenate(neg_ids)
        # self.bpr_dataframe = pd.concat(
        #     [self.bpr_dataframe, twenty_duplicates], ignore_index=True
        # )

    def __len__(self):
        return (
            len(self.bpr_dataframe) if self.set_type == "train" else len(self.dataframe)
        )

    def __getitem__(self, idx):
        # If on training, return BPR samples
        # (user, pos_image, neg_image)
        if self.set_type == "train":
            user_id = self.bpr_dataframe.at[idx, "id_user"]
            pos_image_id = self.bpr_dataframe.at[idx, "id_pos_img"]
            neg_image_id = self.bpr_dataframe.at[idx, "id_neg_img"]
            pos_image = self.datamodule.image_embeddings[pos_image_id]
            neg_image = self.datamodule.image_embeddings[neg_image_id]

            return user_id, pos_image, neg_image

        # If on validation, return samples
        # (id_user, image, label, id_test)
        # The test_id is needed to compute the validation recall or AUC
        # inside the LightningModule
        elif self.set_type == "validation":
            user_id = self.dataframe.at[idx, "id_user"]

            image_id = self.dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])
            test_id = self.dataframe.at[idx, "id_test"]

            return user_id, image, target, test_id

        # If on test, return samples
        # (id_user, image, label)
        elif self.set_type == "test":
            user_id = self.dataframe.at[idx, "id_user"]

            image_id = self.dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            target = self.dataframe.at[idx, self.takeordev].astype(float)

            return user_id, image, target


# Dataset to train with Contrastive Loss criterion
# Compatible with models: COLLEI
class TripadvisorImageAuthorshipCLDataset(TripadvisorImageAuthorshipBCEDataset):
    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipCLDataset, self).__init__(**kwargs)
        self._setup_contrastive_learning_samples()

    # Creates the BPR criterion samples with the form (user, positive image, negative image)
    def _setup_contrastive_learning_samples(self):
        # Filter only positive samples for Contrastive Learning
        self.pos_dataframe = (
            self.dataframe[self.dataframe[self.takeordev] == 1]
            .drop_duplicates(keep="first")
            .reset_index(drop=True)
        )

    def __len__(self):
        return (
            len(self.pos_dataframe) if self.set_type == "train" else len(self.dataframe)
        )

    def __getitem__(self, idx):
        # If on training or validation, return CL samples
        # (user, pos_image)
        if self.set_type == "train":
            user_id = self.pos_dataframe.at[idx, "id_user"]

            image_id = self.pos_dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            return user_id, image

        # If on test, return normal samples
        # (user, image, label)
        elif self.set_type == "validation":
            user_id = self.dataframe.at[idx, "id_user"]

            image_id = self.dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])
            test_id = self.dataframe.at[idx, "id_test"]

            return user_id, image, target, test_id

        elif self.set_type == "test":
            user_id = self.dataframe.at[idx, "id_user"]

            image_id = self.dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])

            return user_id, image, target
