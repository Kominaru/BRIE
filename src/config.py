from argparse import ArgumentParser, BooleanOptionalAction


def read_args():

    parser = ArgumentParser()

    parser.add_argument('--city',  type=str)
    parser.add_argument('--stage',  type=str)
    parser.add_argument('--model',  type=str, nargs='+')
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2**15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-d', type=int, default=256)
    parser.add_argument('--use_train_val', action=BooleanOptionalAction)
    ret_args = parser.parse_args()

    return ret_args
