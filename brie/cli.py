"""Command line interface: ``brie fit`` and ``brie rank``."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from brie.config import int_expr


def _coerce_user(user: str, known_users: list):
    """Command line users are strings; match the type used in training (usually int)."""
    if user in known_users:
        return user
    for cast in (int, float):
        try:
            if cast(user) in known_users:
                return cast(user)
        except ValueError:
            pass
    return user


def _fit(args: argparse.Namespace) -> None:
    from brie.ranker import BRIERanker

    interactions = pd.read_csv(args.interactions)
    features = np.load(args.features)
    ranker = BRIERanker(
        d=args.d,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        accelerator=args.accelerator,
        num_workers=args.workers,
    )
    ranker.fit(
        interactions,
        features,
        user_col=args.user_col,
        image_col=args.image_col,
        item_col=args.item_col,
    )
    ranker.save(args.out)
    print(f"Saved model for {len(ranker.users)} users to {args.out}")


def _rank(args: argparse.Namespace) -> None:
    from brie.ranker import BRIERanker

    ranker = BRIERanker.load(args.model)
    features = np.load(args.features)
    if args.images:
        features = pd.DataFrame(features).loc[args.images]
    user = _coerce_user(args.user, ranker.users)
    ranking = ranker.rank(user, features, top_k=args.top_k)
    if args.output:
        ranking.to_csv(args.output, index=False)
    else:
        ranking.to_csv(sys.stdout, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brie", description="BRIE: personalized image ranking for explainable recommendations"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="train a model on your own data")
    fit.add_argument("--interactions", required=True, help="CSV with one row per image authored by a user")
    fit.add_argument("--features", required=True, help=".npy matrix of image features, row i = image i")
    fit.add_argument("--out", required=True, help="where to save the trained model")
    fit.add_argument("--user-col", default="user")
    fit.add_argument("--image-col", default="image", help="image id = row number in --features")
    fit.add_argument("--item-col", default="item", help="item (e.g. restaurant) each image belongs to")
    fit.add_argument("-d", type=int, default=64, help="latent factors (paper: 64)")
    fit.add_argument("--dropout", type=float, default=0.75)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--epochs", type=int, default=15)
    fit.add_argument("--batch-size", type=int_expr, default=2**14, help="integer or power such as 2**14")
    fit.add_argument("--seed", type=int, default=None)
    fit.add_argument("--accelerator", default="auto")
    fit.add_argument("--workers", type=int, default=0)
    fit.set_defaults(func=_fit)

    rank = sub.add_parser("rank", help="rank images for a user with a trained model")
    rank.add_argument("--model", required=True, help="file written by `brie fit`")
    rank.add_argument("--user", required=True)
    rank.add_argument("--features", required=True, help=".npy matrix of image features, row i = image i")
    rank.add_argument("--images", type=int, nargs="+", help="rows of --features to rank (default: all)")
    rank.add_argument("--top-k", type=int, default=None)
    rank.add_argument("--output", help="write the ranking to this CSV instead of stdout")
    rank.set_defaults(func=_rank)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
