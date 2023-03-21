from argparse import ArgumentParser


def read_args():

    parser = ArgumentParser()

    parser.add_argument('--city', '-C', type=str)
    parser.add_argument('--stage', '-S', type=str)
    parser.add_argument('--model', '-M', type=str, nargs='+')
    parser.add_argument('--workers', '-W', type=int, default=4)
    parser.add_argument('--batch_size', '-B', type=int, default=2**15)
    ret_args = parser.parse_args()

    return ret_args
