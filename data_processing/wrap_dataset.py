
import os
import prody as pr
import sidechainnet as scn
from sidechainnet.utils.measure import get_seq_coords_and_angles
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import pickle
import argparse

from tqdm.contrib.concurrent import process_map

import re
import time

import torch
torch.multiprocessing.set_sharing_strategy('file_system')


def load_complex(path_tuple):
    num, path, complex = path_tuple
    if not os.path.exists(path):
        return None
    filepath = os.path.join(path, 'scn_complex_esm_128.pkl')
    print(f'[{num}] {filepath}')
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as file:
        complex = pickle.load(file)
    return complex


import multiprocessing
import random

if __name__ == '__main__':
    base_path = '../../data/dips_preprocessed/'
    dataset = dict()

    pool = multiprocessing.Pool(processes=40)

    for split in ('DB5', 'train', 'val'):
        split_path = os.path.join(base_path, split)

        paths, counter = [], 0
        for section in os.listdir(split_path):
            section_path = os.path.join(split_path, section)
            for complex in os.listdir(section_path):
                paths.append((counter, os.path.join(section_path, complex), complex))
                counter += 1

        print(f'fetched {len(paths)} paths for {split}')
        random.shuffle(paths)

        paths = paths[:int(0.1 * len(paths))]
        print(f'reduced to {len(paths)} paths for {split}')

        complexes = pool.map(load_complex, paths)
        print(f'fetched {len(complexes)} complexes for {split}')

        split_dataset = defaultdict(list)
        for complex in tqdm(complexes):
            if not complex: continue
            for key in ('ids', 'crd', 'ang', 'seq', 'chn', 'enc', 'tgt_crd'):
                if key not in complex: continue
                split_dataset[key].append(complex[key])

        dataset[split] = split_dataset

    with open('../../data/dips_1024_pruned_esm_128_sample.pkl', 'wb') as file:
        pickle.dump(dataset, file)

    print('done')
