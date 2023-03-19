from argparse import ArgumentParser


def read_args():

    parser = ArgumentParser()

    parser.add_argument('--city', '-C', type=str)
    parser.add_argument('--stage', '-S', type=str)
    parser.add_argument('--model', '-M', type=str, nargs='+')

    ret_args = parser.parse_args()

    return ret_args
