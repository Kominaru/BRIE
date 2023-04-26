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

    def __init__(self, city, batch_size, num_workers=4, dataset_class=None, use_train_val=False) -> None:
        super().__init__()
        self.city = city
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_class = TripadvisorImageAuthorshipBCEDataset if dataset_class is None else dataset_class
        self.use_train_val = use_train_val

        self.setup()

    def setup(self, stage=None):
        self.image_embeddings = Tensor(pickle.load(
            open("C:/Users/Komi/Papers/PRESLEY/data/"+self.city+'/data_10+10/IMG_VEC', 'rb')))

        self.train_dataset = self._get_dataset(
            'TRAIN' if not self.use_train_val else 'TRAIN_DEV')
        self.train_val_dataset = self._get_dataset('TRAIN_DEV')
        self.val_dataset = self._get_dataset(
            'DEV' if not self.use_train_val else 'TEST', set_type='validation')
        self.test_dataset = self._get_dataset('TEST', set_type='test')

        print(
            f"{self.city:<10} | {self.train_dataset.nusers} users | {len(self.image_embeddings)} images")

        self.nusers = self.train_dataset.nusers

    def _get_dataset(self, set_name, set_type='train'):
        return self.dataset_class(datamodule=self, city=self.city, partition_name=set_name, set_type=set_type)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=16384, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=16384, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True)


# Dataset to train with BCE criterion
# Compatible with models: MF_ELVis,  ELVis
class TripadvisorImageAuthorshipBCEDataset(Dataset):

    def __init__(self, datamodule: ImageAuthorshipDataModule, city=None, partition_name=None, set_type='train'):

        self.set_type = set_type
        self.city = city
        self.datamodule = datamodule
        self.partition_name = partition_name

        # Name of the column that indicates sample label varies between partitions
        self.takeordev = 'is_dev' if partition_name in [
            'DEV', 'TEST'] else 'take'

        self.dataframe = pickle.load(
            open(f"C:/Users/Komi/Papers/PRESLEY/data/{city}/data_10+10/{partition_name}_IMG", 'rb'))

        self.nusers = self.dataframe['id_user'].nunique()

        print(f'{self.set_type} partition ({self.partition_name}_IMG)   | {len(self.dataframe)} samples | {self.nusers} users')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = self.dataframe.at[idx, 'id_user']

        image_id = self.dataframe.at[idx, 'id_img']
        image = self.datamodule.image_embeddings[image_id]

        target = float(self.dataframe.at[idx, self.takeordev])

        return user_id, image, target


# Dataset to train with BPR criterion
# Compatible with models: PRESLEY
class TripadvisorImageAuthorshipBPRDataset(TripadvisorImageAuthorshipBCEDataset):

    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipBPRDataset,
              self).__init__(**kwargs)
        self._setup_bpr_dataframe()

    # Creates the BPR criterion samples with the form (user, positive image, negative image)
    def _setup_bpr_dataframe(self):

        # Separate between positive and negative samples
        self.positive_samples = self.dataframe[self.dataframe[self.takeordev] == 1].sort_values([
            'id_user', 'id_img']).rename(columns={'id_img': 'id_pos_img'}).reset_index(drop=True)
        # self.positive_samples = self.positive_samples.drop_duplicates(
        #     keep='first').reset_index(drop=True)

        self._resample_dataframe()

    def _resample_dataframe(self):

        num_samples = len(self.positive_samples)

        same_res_bpr_samples = self.positive_samples.copy()
        different_res_bpr_samples = self.positive_samples.copy()

        # 1. Select 10 images not from U and not from the same restaurant
        user_ids = self.positive_samples['id_user'].to_numpy()[
            :, None]
        img_ids = self.positive_samples['id_pos_img'].to_numpy()[
            :, None]
        rest_ids = self.positive_samples['id_pos_img'].to_numpy()[
            :, None]

        # List of the sample no. of the new neg_img of each BPR sample
        new_negatives = randint(
            num_samples, size=num_samples)

        # Count how many would have the same user in the neg_img and the pos_img
        num_invalid_samples = np.sum(
            ((
                user_ids[new_negatives] == user_ids) | (rest_ids[new_negatives] == rest_ids)))
        while num_invalid_samples > 0:
            # Resample again the neg images for those samples, until all are valid,
            # meaning that user(pos_img(sample)) =/= user(neg_img(sample))
            new_negatives[np.where(((
                user_ids[new_negatives] == user_ids) | (rest_ids[new_negatives] == rest_ids)))[0]] = randint(
                num_samples, size=num_invalid_samples)

            num_invalid_samples = np.sum(((
                user_ids[new_negatives] == user_ids) | (rest_ids[new_negatives] == rest_ids)))

        # Assign as new neg imgs the img_ids of the selected neg_imgs
        different_res_bpr_samples['id_neg_img'] = img_ids[new_negatives]

        # 1. Select 10 images not from U but from the same restaurant as the positive
        def obtain_samerest_samples(rest):

            # Works the same way as the previous algorithm, but only restaurant-wise
            user_ids = rest['id_user'].to_numpy()[:, None]
            img_ids = rest['id_pos_img'].to_numpy()[:, None]

            new_negatives = randint(len(rest), size=len(rest))
            num_invalid_samples = np.sum(user_ids[new_negatives] == user_ids)
            while num_invalid_samples > 0:
                new_negatives[np.where(user_ids[new_negatives] == user_ids)[0]] = randint(
                    len(rest), size=num_invalid_samples)

                num_invalid_samples = np.sum(
                    user_ids[new_negatives] == user_ids)
            rest['id_neg_img'] = img_ids[new_negatives]

            return rest

        # Can't select "same restaurant" negative samples if all that restaurant's photos
        # are by the same user
        same_res_bpr_samples = same_res_bpr_samples.groupby(
            'id_restaurant').filter(lambda g: g['id_user'].nunique() > 1).reset_index(drop=True)
        same_res_bpr_samples = same_res_bpr_samples.groupby(
            'id_restaurant', group_keys=False).apply(obtain_samerest_samples).reset_index(drop=True)

        self.bpr_dataframe = pd.concat(
            [different_res_bpr_samples, same_res_bpr_samples], axis=0, ignore_index=True)

    def __len__(self):
        return len(self.bpr_dataframe) if self.set_type == 'train' else len(self.dataframe)

    def __getitem__(self, idx):
        # If on training, return BPR samples
        # (user, pos_image, neg_image)
        if self.set_type == 'train':

            user_id = self.bpr_dataframe.at[idx, 'id_user']
            pos_image_id = self.bpr_dataframe.at[idx, 'id_pos_img']
            neg_image_id = self.bpr_dataframe.at[idx, 'id_neg_img']
            pos_image = self.datamodule.image_embeddings[pos_image_id]
            neg_image = self.datamodule.image_embeddings[neg_image_id]

            return user_id, pos_image, neg_image

        # If on validation, return samples
        # (id_user, image, label, id_test)
        # The test_id is needed to compute the validation recall or AUC
        # inside the LightningModule
        elif self.set_type == 'validation':
            user_id = self.dataframe.at[idx, 'id_user']

            image_id = self.dataframe.at[idx, 'id_img']
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])
            test_id = self.dataframe.at[idx, 'id_test']

            return user_id, image, target, test_id

        # If on test, return samples
        # (id_user, image, label)
        elif self.set_type == 'test':

            user_id = self.dataframe.at[idx, 'id_user']

            image_id = self.dataframe.at[idx, 'id_img']
            image = self.datamodule.image_embeddings[image_id]

            target = self.dataframe.at[idx, self.takeordev].astype(float)

            return user_id, image, target


# Dataset to train with Contrastive Loss criterion
# Compatible with models: COLLEI
class TripadvisorImageAuthorshipCLDataset(TripadvisorImageAuthorshipBCEDataset):

    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipCLDataset,
              self).__init__(**kwargs)
        self._setup_contrastive_learning_samples()

    # Creates the BPR criterion samples with the form (user, positive image, negative image)
    def _setup_contrastive_learning_samples(self):

        # Filter only positive samples for Contrastive Learning
        self.pos_dataframe = self.dataframe[self.dataframe[self.takeordev] == 1].drop_duplicates(keep='first').reset_index(
            drop=True)

    def __len__(self):
        return len(self.pos_dataframe) if self.set_type == 'train' else len(self.dataframe)

    def __getitem__(self, idx):
        # If on training or validation, return CL samples
        # (user, pos_image)
        if self.set_type == 'train':

            user_id = self.pos_dataframe.at[idx, 'id_user']

            image_id = self.pos_dataframe.at[idx, 'id_img']
            image = self.datamodule.image_embeddings[image_id]

            return user_id, image

        # If on test, return normal samples
        # (user, image, label)
        elif self.set_type == 'validation':

            user_id = self.dataframe.at[idx, 'id_user']

            image_id = self.dataframe.at[idx, 'id_img']
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])
            test_id = self.dataframe.at[idx, 'id_test']

            return user_id, image, target, test_id

        elif self.set_type == 'test':

            user_id = self.dataframe.at[idx, 'id_user']

            image_id = self.dataframe.at[idx, 'id_img']
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])

            return user_id, image, target
