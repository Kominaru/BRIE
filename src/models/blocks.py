from torch import nn, Tensor
import torch

# Embedding Block for image authorship task
# Inputs: user id, pre-trained image embedding
# Outputs (batch_size x d) user embedding, (batch_size x d) image embedding


class ImageAutorshipEmbeddingBlock(nn.Module):
    def __init__(self, d, nusers, initrange=0.05):
        super().__init__()
        self.d = d
        self.nusers = nusers
        self.u_emb = nn.Embedding(num_embeddings=nusers, embedding_dim=d)
        self.img_fc = nn.Linear(1536, d)
        # self._init_weights(initrange=initrange)  # UNCOMMENT THIS TO INIT WEIGHTS

    def _init_weights(self, initrange=0.01):
        self.u_emb.weight.data.uniform_(-initrange, initrange)
        self.img_fc.weight.data.uniform_(-initrange, initrange)
        self.img_fc.bias.data.zero_()

    def forward(self, users, images):
        # Ensure we work with tensors in the case of single sample inputs
        if not torch.is_tensor(users):
            users = torch.tensor(users, dtype=torch.int32)

        u_embeddings = self.u_emb(users)
        img_embeddings = self.img_fc(images)

        return u_embeddings, img_embeddings
