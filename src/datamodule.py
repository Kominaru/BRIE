import pickle
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from lightning.pytorch import LightningDataModule
import pandas as pd

# City-wise Datamodule, contains the image embeddings (common to all partitions)
# and all the required partitions (train, train+val, val, test)


class ImageAuthorshipDataModule(LightningDataModule):

    def __init__(self, city, batch_size, num_workers=4, dataset_class=None) -> None:
        super().__init__()
        self.city = city
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_class = TripadvisorImageAuthorshipBCEDataset if dataset_class is None else dataset_class

        self.setup()

    def setup(self, stage=None):
        self.image_embeddings = Tensor(pickle.load(
            open("data/"+self.city+'/data_10+10/IMG_VEC', 'rb')))

        self.train_dataset = self._get_dataset('TRAIN')
        self.train_val_dataset = self._get_dataset('TRAIN_DEV')
        self.val_dataset = self._get_dataset('DEV')
        self.test_dataset = self._get_dataset('TEST')

        print(
            f"{self.city:<10} | {self.train_dataset.nusers} users | {len(self.image_embeddings)} images")

        self.nusers = self.train_dataset.nusers

    def _get_dataset(self, set_name):
        return self.dataset_class(datamodule=self, city=self.city, partition_name=set_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


# Dataset to train with BCE criterion
# Compatible with models: MF_ELVis,  ELVis
class TripadvisorImageAuthorshipBCEDataset(Dataset):

    def __init__(self, datamodule: ImageAuthorshipDataModule, city=None, partition_name=None):
        self.city = city
        self.datamodule = datamodule
        self.partition_name = partition_name

        # Name of the column that indicates sample label varies between partitions
        self.takeordev = 'is_dev' if partition_name in [
            'DEV', 'TEST'] else 'take'

        self.dataframe = pickle.load(
            open(f"data/{city}/data_10+10/{partition_name}_IMG", 'rb'))

        self.nusers = self.dataframe['id_user'].nunique()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = self.dataframe.loc[idx, 'id_user']

        image_id = self.dataframe.loc[idx, 'id_img']
        image = self.datamodule.image_embeddings[image_id]

        target = self.dataframe.loc[idx, self.takeordev].astype(float)

        return user_id, image, target


# Dataset to train with BPR criterion
# Compatible with models: PRESLEY
class TripadvisorImageAuthorshipBPRDataset(TripadvisorImageAuthorshipBCEDataset):

    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipBPRDataset,
              self).__init__(**kwargs)
        self._setup_bpr_samples()

    # Creates the BPR criterion samples with the form (user, positive image, negative image)
    def _setup_bpr_samples(self):

        # Separate between positive and negative samples
        positive_samples = self.dataframe[self.dataframe[self.takeordev] == 1].sort_values([
            'id_user', 'id_img']).rename(columns={'id_img': 'id_pos_img'}).reset_index(drop=True)
        negative_samples = self.dataframe[self.dataframe[self.takeordev] == 0].sort_values([
            'id_user', 'id_img']).rename(columns={'id_img': 'id_neg_img'}).reset_index(drop=True)

        # If dealing with the original ELVis datasets:

        # TRAIN and TRAIN_DEV have, per each original (user, image, 1) samples, 20 repetitions and 20
        # negative samples (user, image', 0). We therefore can generate len(self.dataframe)/2 BPR samples
        if self.takeordev == 'take':
            self.bpr_dataframe = pd.concat(
                [positive_samples[['id_user', 'id_pos_img']], negative_samples['id_neg_img']], axis=1)

        # DEV and TEST have, per each of the n original  (user, image, 1) sample, a (user, image, 0) sample for all the
        # n' other images of that restaurant present in the training set. Here I choose to reduce them to
        # n BPR samples, but we could create up to n' for each positive one.
        elif self.takeordev == 'is_dev':
            self.bpr_dataframe = pd.merge(
                left=positive_samples[['id_user', 'id_pos_img', 'id_test']
                                      ], right=negative_samples[['id_user', 'id_neg_img']],
                left_on='id_user', right_on='id_user', how='left'
            ).sample(frac=1).drop_duplicates('id_test', keep='first').reset_index(drop=True)

    def __len__(self):
        return len(self.bpr_dataframe) if self.partition_name != 'TEST' else len(self.dataframe)

    def __getitem__(self, idx):
        # If on training or validation, return BPR samples
        # (user, pos_image, neg_image)
        if self.partition_name != 'TEST':

            user_id = self.bpr_dataframe.loc[idx, 'id_user']
            pos_image_id = self.bpr_dataframe.loc[idx, 'id_pos_img']
            neg_image_id = self.bpr_dataframe.loc[idx, 'id_neg_img']
            pos_image = self.datamodule.image_embeddings[pos_image_id]
            neg_image = self.datamodule.image_embeddings[neg_image_id]

            return user_id, pos_image, neg_image

        # If on test, return normal samples
        # (user, image, label)
        elif self.partition_name == 'TEST':

            user_id = self.dataframe.loc[idx, 'id_user']

            image_id = self.dataframe.loc[idx, 'id_img']
            image = self.datamodule.image_embeddings[image_id]

            target = self.dataframe.loc[idx, self.takeordev].astype(float)

            return user_id, image, target
