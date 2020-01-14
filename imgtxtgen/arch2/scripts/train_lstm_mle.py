"""
Train recursive lstm using MLE.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os

from argparse import ArgumentParser
from imgtxtgen.common.datasets.cub2011 import CUB2011Dataset

def parse_args():
    dataset_path = os.path.join('users', 'gpu', 'haribala', 'code', 'datasets', 'CUB_200_2011', 'CUB_200_2011')
    outputs_path = os.path.join('outputs', 'arch1')

    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Path to CUB2011 directory.', default=dataset_path)
    parser.add_argument('--outputs', type=str, help='Directory to store training outputs.', default=outputs_path)

    args = parser.parse_args()

    assert os.path.isdir(args.dataset), f'CUB2011 path is not a directory:{args.dataset}'

    return

if __name__ == '__main__':
    f(parse_args())