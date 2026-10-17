"""End-to-end example on a tiny synthetic dataset: train, rank, save and reload.

    python examples/synthetic_example.py
"""

import numpy as np
import pandas as pd

from brie import BRIERanker

rng = np.random.default_rng(0)
n_users, n_items, images_per_item, n_features = 20, 40, 6, 32

# 1. Image features: one row per image (here random; in practice CNN embeddings).
image_features = rng.normal(size=(n_items * images_per_item, n_features)).astype("float32")

# 2. Interactions: one row per image, with the user that authored it and the item
#    (e.g. the restaurant) it belongs to. The image is the row number in image_features.
interactions = pd.DataFrame(
    {
        "user": rng.integers(n_users, size=len(image_features)),
        "image": np.arange(len(image_features)),
        "item": np.repeat(np.arange(n_items), images_per_item),
    }
)
print(interactions.head())

# 3. Train. The defaults are the paper's hyperparameters.
ranker = BRIERanker(d=16, epochs=5, batch_size=128, seed=0).fit(interactions, image_features)

# 4. Rank the photos of one item for one user.
candidates = image_features[:images_per_item]
print(ranker.rank(user=3, image_features=candidates))

# 5. Save and reload.
ranker.save("brie_example.pt")
reloaded = BRIERanker.load("brie_example.pt")
print(reloaded.rank(user=3, image_features=candidates, top_k=3))
