# Compute the test set predictions based on the centroids of the training set

import numpy as np
import pandas as pd

from src.datamodule import ImageAuthorshipDataModule
from tqdm import tqdm


def get_centroid_preds(dm: ImageAuthorshipDataModule):
    def get_centroid(user_df: pd.DataFrame):
        return np.mean(image_embeddings[user_df["id_img"].values], axis=0)

    train_set = dm.train_dataset.dataframe
    train_set = train_set[train_set["take"] == 1].drop_duplicates(subset=["id_img"])

    test_img_ids = dm.test_dataset.dataframe["id_img"].values
    test_user_ids = dm.test_dataset.dataframe["id_user"].values

    image_embeddings = dm.image_embeddings.cpu().numpy()

    train_centroids = train_set.groupby("id_user").apply(get_centroid)

    test_preds = []

    for i in tqdm(range(test_img_ids.shape[0])):
        img_id = test_img_ids[i]
        user_id = test_user_ids[i]

        test_preds.append(
            1
            / (1 + np.linalg.norm(image_embeddings[img_id] - train_centroids[user_id]))
        )

    return np.array(test_preds)
