import numpy as np
import pandas as pd
import pytest

from brie import BRIERanker
from brie.cli import main


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n_items, per_item, n_users, n_features = 12, 5, 8, 16
    features = rng.normal(size=(n_items * per_item, n_features)).astype("float32")
    interactions = pd.DataFrame(
        {
            "user": rng.choice(list("abcdefgh"), size=len(features)),
            "image": np.arange(len(features)),
            "item": np.repeat(np.arange(n_items), per_item),
        }
    )
    return interactions, features


@pytest.fixture(scope="module")
def ranker(data):
    interactions, features = data
    return BRIERanker(d=8, epochs=2, batch_size=32, seed=0).fit(interactions, features, verbose=False)


def test_rank_orders_by_score(ranker, data):
    _, features = data
    ranking = ranker.rank("a", features[:5])
    assert list(ranking.columns) == ["image", "score"]
    assert sorted(ranking["image"]) == list(range(5))
    assert ranking["score"].is_monotonic_decreasing
    assert ranker.rank("a", features[:5], top_k=2).shape[0] == 2


def test_rank_uses_dataframe_index_as_image_id(ranker, data):
    _, features = data
    table = pd.DataFrame(features[:4], index=["w", "x", "y", "z"])
    assert set(ranker.rank("b", table)["image"]) == {"w", "x", "y", "z"}


def test_save_load_roundtrip(ranker, data, tmp_path):
    _, features = data
    ranker.save(tmp_path / "m.pt")
    loaded = BRIERanker.load(tmp_path / "m.pt")
    np.testing.assert_allclose(ranker.score("c", features[:6]), loaded.score("c", features[:6]), rtol=1e-6)
    assert loaded.users == ranker.users


def test_unknown_user_and_bad_features(ranker, data):
    _, features = data
    with pytest.raises(KeyError):
        ranker.rank("nobody", features[:3])
    with pytest.raises(ValueError):
        ranker.rank("a", features[:3, :5])


def test_fit_validates_input(data):
    interactions, features = data
    with pytest.raises(ValueError, match="without features"):
        BRIERanker(epochs=1).fit(interactions, features[:10], verbose=False)
    with pytest.raises(ValueError, match="not found"):
        BRIERanker(epochs=1).fit(interactions.rename(columns={"item": "restaurant"}), features)


def test_cli_fit_and_rank(data, tmp_path, capsys):
    interactions, features = data
    interactions.to_csv(tmp_path / "interactions.csv", index=False)
    np.save(tmp_path / "features.npy", features)
    main(["fit", "--interactions", str(tmp_path / "interactions.csv"), "--features", str(tmp_path / "features.npy"),
          "--out", str(tmp_path / "m.pt"), "-d", "8", "--epochs", "1", "--batch-size", "2**5", "--seed", "0"])
    main(["rank", "--model", str(tmp_path / "m.pt"), "--user", "a",
          "--features", str(tmp_path / "features.npy"), "--images", "0", "1", "2", "--top-k", "2"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-3] == "image,score" and len(out) >= 3
