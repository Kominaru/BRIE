import pickle
from torch.utils.data import Dataset
from torch import Tensor
# City-wise dataset, contains the image embeddings (common to all partitions)
# and all the required partitions (train, train+val, val, test)


class Tripadvisor_ImageAuthorship_Dataset():
    def __init__(self, city=None):
        self.city = city
        self.image_embeddings = Tensor(pickle.load(
            open("data/"+city+'/data_10+10/IMG_VEC', 'rb')))

        
        self.train_data = self.create_partition(city, 'TRAIN')
        self.train_val_data = self.create_partition(city, 'TRAIN_DEV')
        self.val_data = self.create_partition(city, 'DEV')
        self.test_data = self.create_partition(city, 'TEST')

        print(f"{city} | {self.train_data.nusers} users | {len(self.image_embeddings)} images")
        
    # Trying to share image_embeddings between all partitions to avoid
    # storing in memory 4 times the large image embedding array
    def create_partition(self, city, set):
        return Tripadvisor_ImageAuthorship_Dataset.Tripadvisor_ImageAuthorship_Set(self, city, set)

    # Single partition
    class Tripadvisor_ImageAuthorship_Set(Dataset):
        def __init__(self, dataset, city=None, set=None):
            
            self.dataset = dataset
            self.samples = pickle.load(
                open("data/"+city+'/data_10+10/'+set+'_IMG', 'rb'))

            self.nusers = self.samples['id_user'].nunique()
            self.city = city

            # Name of the class label column
            self.takeordev = 'is_dev' if set in ['DEV', 'TEST'] else 'take'

            print(f"{set}\t| {self.__len__()} samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            user_id = self.samples.loc[idx, 'id_user']

            image_id = self.samples.loc[idx, 'id_img']
            image = self.dataset.image_embeddings[image_id]

            target = self.samples.loc[idx, self.takeordev].astype(float)

            return user_id, image, target
