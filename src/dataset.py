import pickle
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from lightning.pytorch import LightningDataModule

# City-wise dataset, contains the image embeddings (common to all partitions)
# and all the required partitions (train, train+val, val, test)        
class ImageAuthorshipDataModule(LightningDataModule):

    def __init__(self,city,batch_size,num_workers=4) -> None:
        super().__init__()
        self.city = city
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.image_embeddings = Tensor(pickle.load(
            open("data/"+self.city+'/data_10+10/IMG_VEC', 'rb')))
    
        self.train_dataset = TripadvisorImageAuthorshipDataset(self,self.city,'TRAIN')

        print(f"{self.city:<10} | {self.train_dataset.nusers} users | {len(self.image_embeddings)} images")

        self.nusers = self.train_dataset.nusers
    
        self.train_val_dataset = TripadvisorImageAuthorshipDataset(self,self.city,'TRAIN_DEV')  
        self.val_dataset = TripadvisorImageAuthorshipDataset(self,self.city,'DEV')
        self.test_dataset = TripadvisorImageAuthorshipDataset(self,self.city,'TEST')


        

    def train_dataloader(self):
        return DataLoader(self.train_val_dataset,batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers)
        
class TripadvisorImageAuthorshipDataset(Dataset):

    def __init__(self, datamodule: ImageAuthorshipDataModule, city=None, partition_name=None):
        self.city = city
        self.datamodule = datamodule
        # Name of the sample label column 
        self.takeordev = 'is_dev' if partition_name in ['DEV', 'TEST'] else 'take'

        self.dataframe = pickle.load(
            open(f"data/{city}/data_10+10/{partition_name}_IMG",'rb'))

        self.nusers = self.dataframe['id_user'].nunique() 

        # print(f"{partition_name:<10} | {self.__len__()} samples")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = self.dataframe.loc[idx, 'id_user']

        image_id = self.dataframe.loc[idx, 'id_img']
        image = self.datamodule.image_embeddings[image_id]

        target = self.dataframe.loc[idx, self.takeordev].astype(float)

        return user_id, image, target