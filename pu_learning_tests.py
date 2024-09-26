import numpy as np
import pandas as pd
import pickle as pkl

train_dev_df = pkl.load(open("data/barcelona/data_10+10/TRAIN_DEV_IMG", "rb"))

train_dev_df = train_dev_df[train_dev_df["take"] == 1]
train_dev_df = train_dev_df.drop_duplicates(keep="first")

img_embeds = pkl.load(open("data/barcelona/data_10+10/IMG_VEC", "rb"))

user_centroids = np.stack(
    train_dev_df.groupby("id_user")
    .apply(lambda x: img_embeds[x["id_img"]].mean(axis=0))
    .values
)

p90 = train_dev_df.groupby("id_user").apply(
    lambda x: np.percentile(
        np.linalg.norm(img_embeds[x["id_img"]] - user_centroids[x.name], axis=1, ord=2),
        90,
    )
)

img_ids = train_dev_df["id_img"].values
user_ids = train_dev_df["id_user"].values

rand_negs = np.random.randint(len(img_ids), size=len(img_ids))

invalid_negs = np.where(
    (
        np.linalg.norm(
            img_embeds[img_ids[rand_negs]] - user_centroids[user_ids], axis=1, ord=2
        )
        < p90[user_ids]
    )
    | (user_ids == user_ids[rand_negs])
)[0]

i = 0
while len(invalid_negs) > 0 and i < 10:
    rand_negs[invalid_negs] = np.random.randint(len(img_ids), size=len(invalid_negs))
    invalid_negs = np.where(
        (
            np.linalg.norm(
                img_embeds[img_ids[rand_negs]] - user_centroids[user_ids], axis=1, ord=2
            )
            < p90[user_ids]
        )
        | (user_ids == user_ids[rand_negs])
    )[0]

    print(len(invalid_negs))
    i += 1

print("Done")
