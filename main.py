import pytorch_lightning as pl
import torch
from torch import nn, optim
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import pickle

torch.set_float32_matmul_precision('medium')

class Tripadvisor_ImageAuthorship_Dataset(Dataset):
    def __init__(self):
        self.img_embeddings = pickle.load(open("data/"+'barcelona'+'/data_10+10/IMG_VEC','rb'))
        self.samples = pickle.load(open("data/"+'barcelona'+'/data_10+10/TRAIN_IMG','rb'))
        self.nusers = self.samples['id_user'].nunique()

        print(self.samples)
        print(self.nusers)
        print(self.img_embeddings[:5])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        user_id = self.samples.loc[idx,'id_user']

        image_id = self.samples.loc[idx,'id_img']
        image = self.img_embeddings[image_id]

        target = self.samples.loc[idx,'take'].astype(float)

        return user_id, image, target
    

    
dataset = Tripadvisor_ImageAuthorship_Dataset()
dataloader = DataLoader(dataset,2**15,shuffle=True)


class MF_ELVis(pl.LightningModule):
    def __init__(self, d, nusers):
        super().__init__()
        self.d = d
        self.nusers = nusers
        self.u_emb = nn.Embedding(num_embeddings=nusers, embedding_dim=d)
        self.img_fc = nn.Linear(1536,d)

    def training_step(self, batch, batch_idx):
        users, images, targets = batch
        u_embeddings = self.u_emb(users)
        img_embeddings = self.img_fc(images)

        preds = torch.sum(u_embeddings*img_embeddings,dim=1)

        loss = nn.functional.binary_cross_entropy_with_logits(preds,targets)
        accuracy = torchmetrics.functional.accuracy(preds,targets,'binary')
        
        self.log('train_accuracy',accuracy,on_step=False,on_epoch=True,prog_bar=True)
        self.log('train_loss',loss,on_step=False,on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=5e-4)
        return optimizer

model = MF_ELVis(256,dataset.nusers)

trainer = pl.Trainer(max_epochs=50,accelerator='gpu',devices=[0])
trainer.fit(model=model, train_dataloaders=dataloader)
    


        