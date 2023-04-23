from argparse import ArgumentParser, BooleanOptionalAction


def read_args():

    parser = ArgumentParser()

    # Required args #
    parser.add_argument('--city',  type=str)
    parser.add_argument('--stage',  type=str)
    parser.add_argument('--model',  type=str, nargs='+')

    # Used in *train* and *tune* modes #

    parser.add_argument('--batch_size', type=int, default=2**15)
    parser.add_argument('--max_epochs', type=int, default=100)

    # Currently only in *train* #

    parser.add_argument('--lr', type=float, default=1e-3)  # learning rate
    # number of latent factors
    parser.add_argument('-d', type=int, default=256)
    # only do train, no validation
    parser.add_argument('--no_validation', action=BooleanOptionalAction)

    # *Tune* args #
    parser.add_argument('--num_models', type=int, default=100)

    # Only in PRESLEY #
    parser.add_argument('--dropout', type=float, default=0)

    # Optional in all execution modes #

    # Whether to use TRAIN_IMG or TRAIN_DEV_IMG as training set
    parser.add_argument('--use_train_val', action=BooleanOptionalAction)
    parser.add_argument('--workers',  type=int, default=4)

    ret_args = parser.parse_args()

    return ret_args
