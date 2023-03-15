from torch import nn

# Embedding Block for image authorship task
# Inputs: user id, pre-trained image embedding
# Outputs (batch_size x d) user embedding, (batch_size x d) image embedding
class ImageAutorshipEmbeddingBlock(nn.Module):
    def __init__(self, d, nusers):
        super().__init__()
        self.d = d
        self.nusers = nusers
        self.u_emb = nn.Embedding(num_embeddings=nusers, embedding_dim=d)
        self.img_fc = nn.Linear(1536, d)

    def forward(self,users,images):
        u_embeddings = self.u_emb(users)
        img_embeddings = self.img_fc(images)

        return u_embeddings, img_embeddings
